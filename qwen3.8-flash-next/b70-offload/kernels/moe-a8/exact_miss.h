// Exact decode miss path for the NVMe expert tier (docs/exact-miss.md).
//
// Shared by the SYCL library (device kernels in exl3_moe.a8.sycl under EXL3_EXACT_MISS) and the host service thread
// below (pure C++17 + Linux AIO; no SYCL, so it builds and is tested on its own: tests/exact_svc_test.cpp).
//
// Protocol (one control region of CTL_BYTES in USM host memory, mapped by the GPU over PCIe):
//   device (ensure kernel, 1 work-item)  : picks in neither VRAM nor RAM -> job j = seq % J: keys, layer, n, then
//                                          release fence (system) and job.seq = seq (system-scope store)
//   host (ExactSvc thread)               : sees job.seq change, for each key: a ready prefetch page, an in-flight
//                                          prefetch page, or an O_DIRECT read into the job's own stage page i;
//                                          writes addr[j][i]; when all reads landed: done[j] = {status, seq}
//   device (wait kernel, 1 work-item)    : spins until done[j].seq == seq (bounded), patches ptrs_all[key] = addr;
//                                          on timeout/error masks those picks (old a6 behaviour) and rolls back
//                                          the VRAM slot it had assigned, so nothing unread is ever committed
//   device (release kernel, after combine): restores ptrs of keys that got no VRAM slot, rel[j] = seq
//   host                                 : a prefetch page pinned by job seq is reusable once rel[j] >= seq;
//                                          a job slot's demand pages are reused only after its own reads finished
//   hints (optional, next-layer prediction): device appends predicted non-resident keys to a ring, host prefetches
//                                          them into an LRU pool of pages (no device ever reads an unpinned page:
//                                          addresses reach the device only through a job's addr[] answer)
#pragma once
#include <cstdint>

namespace exact {
constexpr int KMAX = 64;                 // max distinct missed experts per MoE call (graph decode M<=4: <= 40)
constexpr int JMAX = 4;                  // max job slots
constexpr int JOB_OFF = 0, JOB_BYTES = 512;          // int32 [seq, layer, n, pad, key[KMAX]]
constexpr int DONE_OFF = 4096, DONE_BYTES = 64;      // int32 [seq, status]          (host writes)
constexpr int REL_OFF = 8192, REL_BYTES = 64;        // int32 [seq]                  (device writes)
constexpr int ADDR_OFF = 12288;                      // int64 [JMAX][KMAX]           (host writes)
constexpr int HINT_OFF = 16384, HINT_CAP = 1024;     // int32 [count, pad x15, key[HINT_CAP]] (device writes)
constexpr int HSTAT_OFF = 24576, HSTAT_N = 24;       // int64 host statistics        (host writes)
constexpr int HLOG_OFF = 26624, LOG_N = 128;         // host job log: int64 [LOG_N][4] = seq, n, t_seen_ns, t_done_ns
constexpr int CTL_BYTES = 32768;
static_assert(ADDR_OFF + JMAX * KMAX * 8 <= HINT_OFF, "ctl layout");
static_assert(HINT_OFF + 64 + HINT_CAP * 4 <= HSTAT_OFF, "ctl layout");
static_assert(HSTAT_OFF + HSTAT_N * 8 <= HLOG_OFF, "ctl layout");
static_assert(HLOG_OFF + LOG_N * 32 <= CTL_BYTES, "ctl layout");
// device scalar block (int64, device memory): see moe_exact_setup
enum XDev : int {
    X_SEQ = 0, X_J = 1, X_N = 2, X_LAYER = 3, X_JOBS = 4, X_NKEYS = 5, X_TIMEOUTS = 6, X_TIMEOUT_KEYS = 7,
    X_OVERFLOW_KEYS = 8, X_HINT_CALLS = 9, X_HINTS = 10, X_HINT_CNT = 11,
    X_POLL_MODE = 12,     // read by the wait kernel on every call (not baked into the graph): see moe_forward_cached
    X_POLLS = 13, X_POLLS_MAX = 14,
    X_KEYS = 16, X_ADDR = 16 + KMAX,
    X_LOG = 16 + 2 * KMAX,   // device job log: int64 [LOG_N][4] = seq, n | outcome << 16, polls, last done value seen
    X_SIZE = 16 + 2 * KMAX + LOG_N * 4
};
// host statistics (HSTAT_OFF, int64)
enum HStat : int {
    H_HEARTBEAT = 0, H_JOBS = 1, H_KEYS = 2, H_DEMAND_READS = 3, H_PF_HIT_READY = 4, H_PF_HIT_INFLIGHT = 5,
    H_PF_READS = 6, H_HINTS = 7, H_HINTS_DROPPED = 8, H_READ_ERRORS = 9, H_SVC_NS = 10, H_SVC_NS_MAX = 11,
    H_BYTES = 12, H_EAGAIN = 13, H_HINTS_KNOWN = 14, H_JOB_DEFERRED = 15, H_STOPPED = 16
};
}  // namespace exact

#if defined(__linux__) || defined(__SYCL_DEVICE_ONLY__)   // host-only code, parsed (not device-compiled) in the SYCL device pass like PfWorker
#include <linux/aio_abi.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <time.h>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace exact {

struct SvcCfg {
    uint8_t* ctl = nullptr;              // CTL_BYTES, shared with the device
    int jobs = 2, kmax = 40;
    int fd = -1;                         // NVMe store, O_RDONLY | O_DIRECT
    int64_t rec = 0;                     // record bytes (4K multiple, <= stride)
    int E = 512;
    std::vector<int64_t> ckpt;           // tier layer li -> store layer (record index = ckpt[li] * E + e)
    uint8_t* stage = nullptr;            // demand pages: page (j, i) at stage + (j * kmax + i) * stride
    uint8_t* pf = nullptr;               // prefetch pool: page p at pf + p * stride
    int pf_pages = 0;
    int64_t stride = 2 << 20;
    int spin_us = 2000;                  // busy-poll this long after the last activity, then sleep
    int sleep_us = 50;
};

class ExactSvc {
public:
    explicit ExactSvc(const SvcCfg& c) : c_(c) {
        if (!c_.ctl || c_.fd < 0 || c_.rec <= 0 || c_.rec > c_.stride || c_.jobs < 1 || c_.jobs > JMAX ||
            c_.kmax < 1 || c_.kmax > KMAX || !c_.stage)
            throw std::runtime_error("ExactSvc: bad config");
        if (syscall(SYS_io_setup, kAio, &ctx_) != 0) throw std::runtime_error(std::string("io_setup: ") + strerror(errno));
        slot_.resize(c_.jobs);
        for (int j = 0; j < c_.jobs; ++j) slot_[j].last_seen = ld(job(j));     // ignore whatever is there at start
        hint_seen_ = ld(hint());
        pf_key_.assign(c_.pf_pages, -1); pf_state_.assign(c_.pf_pages, kFree); pf_last_.assign(c_.pf_pages, 0);
        pf_pin_j_.assign(c_.pf_pages, -1); pf_pin_seq_.assign(c_.pf_pages, 0); pf_wait_.resize(c_.pf_pages);
        th_ = std::thread([this] { run(); });
    }
    ~ExactSvc() { stop(); }
    void stop() {
        if (th_.joinable()) { stop_.store(true); th_.join(); }
        if (ctx_) { syscall(SYS_io_destroy, ctx_); ctx_ = 0; }
    }
    void pause(bool p) { pause_.store(p); }         // tests: stop answering (device must time out, then recover)
    int64_t stat(int i) const { return __atomic_load_n(hs() + i, __ATOMIC_RELAXED); }

private:
    static constexpr int kAio = 512;
    enum : uint8_t { kFree = 0, kLoading = 1, kReady = 2 };
    struct Slot {
        int last_seen = 0, seq = 0, n = 0, outstanding = 0, status = 0;
        bool busy = false;
        int64_t t0 = 0;
    };
    SvcCfg c_;
    aio_context_t ctx_ = 0;
    std::thread th_;
    std::atomic<bool> stop_{false}, pause_{false};
    std::vector<Slot> slot_;
    int hint_seen_ = 0;
    int64_t tick_ = 0;
    std::vector<int> pf_key_; std::vector<uint8_t> pf_state_; std::vector<int64_t> pf_last_;
    std::vector<int> pf_pin_j_, pf_pin_seq_;
    std::vector<std::vector<int>> pf_wait_;          // job slots waiting on a loading prefetch page
    std::unordered_map<int, int> pf_of_;             // key -> page
    int pf_next_ = 0;

    int* job(int j) const { return reinterpret_cast<int*>(c_.ctl + JOB_OFF + j * JOB_BYTES); }
    int* done(int j) const { return reinterpret_cast<int*>(c_.ctl + DONE_OFF + j * DONE_BYTES); }
    int* rel(int j) const { return reinterpret_cast<int*>(c_.ctl + REL_OFF + j * REL_BYTES); }
    int64_t* addr(int j) const { return reinterpret_cast<int64_t*>(c_.ctl + ADDR_OFF) + j * KMAX; }
    int* hint() const { return reinterpret_cast<int*>(c_.ctl + HINT_OFF); }
    int64_t* hs() const { return reinterpret_cast<int64_t*>(c_.ctl + HSTAT_OFF); }
    static int ld(const int* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
    void add(int i, int64_t v) { __atomic_store_n(hs() + i, hs()[i] + v, __ATOMIC_RELAXED); }   // single writer
    static int64_t now_ns() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1000000000LL + t.tv_nsec; }
    uint8_t* stage_page(int j, int i) const { return c_.stage + ((int64_t)j * c_.kmax + i) * c_.stride; }
    uint8_t* pf_page(int p) const { return c_.pf + (int64_t)p * c_.stride; }
    int64_t offset(int key) const {
        int li = key / c_.E, e = key % c_.E;
        return (c_.ckpt.empty() ? li : c_.ckpt.at(li)) * (int64_t)c_.E * c_.rec + (int64_t)e * c_.rec;
    }

    // aio_data: bit 63 = prefetch, low bits = page (prefetch) or slot (demand)
    void submit(std::vector<iocb>& cbs) {
        std::vector<iocb*> ptr(cbs.size());
        for (size_t i = 0; i < cbs.size(); ++i) ptr[i] = &cbs[i];
        size_t off = 0;
        while (off < ptr.size()) {
            long r = syscall(SYS_io_submit, ctx_, (long)(ptr.size() - off), ptr.data() + off);
            if (r > 0) { off += r; continue; }
            if (r < 0 && errno == EAGAIN) { add(H_EAGAIN, 1); reap(1, 100000000L); continue; }
            // hard submit error: fail the remaining requests through the normal completion path
            for (size_t i = off; i < ptr.size(); ++i) complete(ptr[i]->aio_data, -1);
            break;
        }
    }
    static iocb make(int fd, void* dst, int64_t n, int64_t off, uint64_t data) {
        iocb cb; memset(&cb, 0, sizeof cb);
        cb.aio_lio_opcode = IOCB_CMD_PREAD; cb.aio_fildes = (uint32_t)fd;
        cb.aio_buf = (uint64_t)(uintptr_t)dst; cb.aio_nbytes = (uint64_t)n; cb.aio_offset = off; cb.aio_data = data;
        return cb;
    }
    int reap(int min_nr, long timeout_ns = 0) {
        io_event ev[64];
        timespec to{timeout_ns / 1000000000L, timeout_ns % 1000000000L};
        long r = syscall(SYS_io_getevents, ctx_, (long)min_nr, 64L, ev, (min_nr && timeout_ns <= 0) ? nullptr : &to);
        for (long i = 0; i < r; ++i) complete(ev[i].data, ev[i].res);
        return r > 0 ? (int)r : 0;
    }
    void complete(uint64_t data, int64_t res) {
        bool ok = res == c_.rec;
        if (!ok) add(H_READ_ERRORS, 1); else add(H_BYTES, c_.rec);
        if (data >> 63) {
            int p = (int)(data & 0xffffffff);
            pf_state_[p] = ok ? kReady : kFree;
            if (!ok) { pf_of_.erase(pf_key_[p]); pf_key_[p] = -1; }
            for (int j : pf_wait_[p]) { if (!ok) slot_[j].status = 1; finish_one(j); }
            pf_wait_[p].clear();
        } else {
            int j = (int)(data & 0xffffffff);
            if (!ok) slot_[j].status = 1;
            finish_one(j);
        }
    }
    void finish_one(int j) {
        Slot& s = slot_[j];
        if (--s.outstanding == 0) answer(j);
    }
    int64_t* hlog(int seq) const { return reinterpret_cast<int64_t*>(c_.ctl + HLOG_OFF) + ((unsigned)seq % LOG_N) * 4; }
    void answer(int j) {
        Slot& s = slot_[j];
        int* d = done(j);
        int64_t t1 = now_ns(), dt = t1 - s.t0;
        __atomic_store_n(hlog(s.seq) + 3, t1, __ATOMIC_RELAXED);   // debug log first: complete once done is visible
        __atomic_store_n(d + 1, s.status, __ATOMIC_RELAXED);
        __atomic_thread_fence(__ATOMIC_RELEASE);          // addr[] and status before the seq the device polls
        __atomic_store_n(d, s.seq, __ATOMIC_RELEASE);
        s.busy = false;
        add(H_SVC_NS, dt);
        if (dt > hs()[H_SVC_NS_MAX]) __atomic_store_n(hs() + H_SVC_NS_MAX, dt, __ATOMIC_RELAXED);
    }
    bool pf_pinned(int p) const {
        int j = pf_pin_j_[p];
        return j >= 0 && (int)(ld(rel(j)) - pf_pin_seq_[p]) < 0;
    }
    bool poll_jobs() {
        bool any = false;
        for (int j = 0; j < c_.jobs; ++j) {
            Slot& s = slot_[j];
            if (s.busy) continue;                         // this slot's reads still in flight: its pages are not free
            int seq = ld(job(j));
            if (seq == s.last_seen) continue;
            any = true;
            s.last_seen = seq;
            const int* jb = job(j);
            int n = __atomic_load_n(jb + 2, __ATOMIC_RELAXED);
            if (n < 0 || n > c_.kmax) n = 0;
            s.seq = seq; s.n = n; s.status = 0; s.outstanding = 1; s.busy = true; s.t0 = now_ns();
            {   int64_t* lg = hlog(seq);
                __atomic_store_n(lg + 3, (int64_t)0, __ATOMIC_RELAXED); __atomic_store_n(lg + 2, s.t0, __ATOMIC_RELAXED);
                __atomic_store_n(lg + 1, (int64_t)n, __ATOMIC_RELAXED); __atomic_store_n(lg, (int64_t)seq, __ATOMIC_RELAXED); }
            add(H_JOBS, 1); add(H_KEYS, n);
            std::vector<iocb> cbs;
            int64_t* a = addr(j);
            for (int i = 0; i < n; ++i) {
                int key = __atomic_load_n(jb + 4 + i, __ATOMIC_RELAXED);
                auto it = pf_of_.find(key);
                if (it != pf_of_.end()) {
                    int p = it->second;
                    a[i] = (int64_t)(uintptr_t)pf_page(p);
                    pf_pin_j_[p] = j; pf_pin_seq_[p] = seq; pf_last_[p] = ++tick_;
                    if (pf_state_[p] == kReady) { add(H_PF_HIT_READY, 1); continue; }
                    add(H_PF_HIT_INFLIGHT, 1);
                    pf_wait_[p].push_back(j); ++s.outstanding;
                    continue;
                }
                uint8_t* dst = stage_page(j, i);
                a[i] = (int64_t)(uintptr_t)dst;
                cbs.push_back(make(c_.fd, dst, c_.rec, offset(key), (uint64_t)j));
                ++s.outstanding;
                add(H_DEMAND_READS, 1);
            }
            if (!cbs.empty()) submit(cbs);
            finish_one(j);                                // drop the planning reference (answers if nothing pending)
        }
        return any;
    }
    int pf_victim() {
        // a free page first, else the least recently used ready page that no unreleased job references
        int best = -1; int64_t bl = INT64_MAX;
        for (int k = 0; k < c_.pf_pages; ++k) {
            int p = (pf_next_ + k) % c_.pf_pages;
            if (pf_state_[p] == kFree) { pf_next_ = (p + 1) % c_.pf_pages; return p; }
            if (pf_state_[p] == kReady && pf_last_[p] < bl && !pf_pinned(p)) { bl = pf_last_[p]; best = p; }
        }
        return best;
    }
    bool poll_hints() {
        int cnt = ld(hint());
        if (cnt == hint_seen_) return false;
        int n = cnt - hint_seen_;
        if (n < 0 || n > HINT_CAP) { add(H_HINTS_DROPPED, n < 0 ? 0 : n - HINT_CAP); n = n < 0 ? 0 : HINT_CAP; }
        const int* ring = hint() + 16;
        std::vector<iocb> cbs;
        for (int k = cnt - n; k != cnt; ++k) {
            int key = __atomic_load_n(ring + ((unsigned)k % HINT_CAP), __ATOMIC_RELAXED);
            add(H_HINTS, 1);
            auto it = pf_of_.find(key);
            if (it != pf_of_.end()) { pf_last_[it->second] = ++tick_; add(H_HINTS_KNOWN, 1); continue; }
            if (c_.pf_pages == 0) { add(H_HINTS_DROPPED, 1); continue; }
            int p = pf_victim();
            if (p < 0) { add(H_HINTS_DROPPED, 1); continue; }
            if (pf_key_[p] >= 0) pf_of_.erase(pf_key_[p]);
            pf_key_[p] = key; pf_state_[p] = kLoading; pf_last_[p] = ++tick_; pf_pin_j_[p] = -1;
            pf_of_[key] = p;
            cbs.push_back(make(c_.fd, pf_page(p), c_.rec, offset(key), (1ULL << 63) | (uint64_t)p));
            add(H_PF_READS, 1);
        }
        hint_seen_ = cnt;
        if (!cbs.empty()) submit(cbs);
        return true;
    }
    void run() {
        int64_t idle_since = now_ns();
        while (!stop_.load(std::memory_order_relaxed)) {
            bool any = false;
            if (!pause_.load(std::memory_order_relaxed)) {
                any |= reap(0) > 0;
                any |= poll_hints();                     // hints first: a demand job for the same key then hits it
                any |= poll_jobs();
            }
            __atomic_store_n(hs() + H_HEARTBEAT, now_ns(), __ATOMIC_RELAXED);
            int64_t t = now_ns();
            if (any) { idle_since = t; continue; }
            bool inflight = false;
            for (auto& s : slot_) inflight |= s.busy;
            if (inflight || t - idle_since < (int64_t)c_.spin_us * 1000) {
#if defined(__x86_64__)
                __builtin_ia32_pause();
#endif
                continue;
            }
            timespec ts{0, (long)c_.sleep_us * 1000};
            nanosleep(&ts, nullptr);
        }
        while (true) {                                    // drain in-flight reads before the pages go away
            bool inflight = false;
            for (auto& s : slot_) inflight |= s.busy;
            for (auto st : pf_state_) inflight |= st == kLoading;
            if (!inflight || reap(1, 1000000000L) == 0) break;   // bounded: never hang shutdown
        }
        __atomic_store_n(hs() + H_STOPPED, 1, __ATOMIC_RELAXED);
    }
};

}  // namespace exact
#endif
