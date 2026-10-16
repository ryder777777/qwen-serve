// CPU-only test of the exact-miss host service (exact_miss.h): a thread plays the device side of the protocol
// (jobs, hints, bounded spin-wait, release) against a real file read with Linux AIO (O_DIRECT when the filesystem
// allows it). Checks: every answered address holds exactly the requested expert's record, prefetch hits happen,
// a paused service makes the "device" time out without corrupting any later answer.
//   g++ -O2 -std=c++17 -pthread -I.. exact_svc_test.cpp -o exact_svc_test && ./exact_svc_test /tmp/store.bin
#include "../exact_miss.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>

using namespace exact;

static const int L = 6, E = 64;
static const int64_t REC = 8192, STRIDE = 8192;
static int ckpt_of(int li) { return L - 1 - li; }                 // tier layer -> store layer (reversed on purpose)
static uint64_t word(int store_layer, int e, int w) { return (uint64_t)(store_layer * E + e) * 1000003ULL + w; }

static int64_t now_ns() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1000000000LL + t.tv_nsec; }

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/tmp/exact_store.bin";
    {   // store: L*E records, record (sl, e) filled with word(sl, e, i)
        FILE* f = fopen(path, "wb");
        std::vector<uint64_t> buf(REC / 8);
        for (int sl = 0; sl < L; ++sl)
            for (int e = 0; e < E; ++e) {
                for (size_t i = 0; i < buf.size(); ++i) buf[i] = word(sl, e, (int)i);
                fwrite(buf.data(), 1, REC, f);
            }
        fclose(f);
    }
    int fd = open(path, O_RDONLY | O_DIRECT);
    bool direct = fd >= 0;
    if (fd < 0) fd = open(path, O_RDONLY);
    printf("store %s, O_DIRECT %s\n", path, direct ? "yes" : "no (fs refused, plain reads)");

    SvcCfg c;
    c.ctl = (uint8_t*)aligned_alloc(4096, CTL_BYTES); memset(c.ctl, 0, CTL_BYTES);
    c.jobs = 2; c.kmax = 40; c.fd = fd; c.rec = REC; c.E = E; c.stride = STRIDE;
    for (int li = 0; li < L; ++li) c.ckpt.push_back(ckpt_of(li));
    c.stage = (uint8_t*)aligned_alloc(4096, (size_t)c.jobs * c.kmax * STRIDE);
    c.pf_pages = 32;
    c.pf = (uint8_t*)aligned_alloc(4096, (size_t)c.pf_pages * STRIDE);
    c.spin_us = 500; c.sleep_us = 20;
    ExactSvc svc(c);

    auto job = [&](int j) { return reinterpret_cast<int*>(c.ctl + JOB_OFF + j * JOB_BYTES); };
    auto done = [&](int j) { return reinterpret_cast<int*>(c.ctl + DONE_OFF + j * DONE_BYTES); };
    auto rel = [&](int j) { return reinterpret_cast<int*>(c.ctl + REL_OFF + j * REL_BYTES); };
    auto addr = [&](int j) { return reinterpret_cast<int64_t*>(c.ctl + ADDR_OFF) + j * KMAX; };
    int* hint = reinterpret_cast<int*>(c.ctl + HINT_OFF);

    std::mt19937 rng(1234);
    int seq = 0, hint_cnt = 0;
    int64_t ok_jobs = 0, bad_pages = 0, timeouts = 0, unexpected_timeouts = 0, keys_checked = 0;
    int64_t wait_ns_total = 0, wait_ns_max = 0, bad_log = 0;
    const int ITERS = 4000;
    std::vector<int> next_keys;
    for (int it = 0; it < ITERS; ++it) {
        bool paused = it >= 1500 && it < 1520;            // service stops answering for 20 jobs
        svc.pause(paused);
        int layer = it % L;
        int n = (int)(rng() % 11);                        // 0..10 distinct misses (decode M=1: <= 10)
        if (it % 50 == 0) n = c.kmax;                     // worst case: M=4, 40 distinct
        std::set<int> ks;
        // half of the time the previous step "predicted" some of these keys (hints -> prefetch pool)
        for (int k : next_keys) if ((int)ks.size() < n) ks.insert(k);
        while ((int)ks.size() < n) ks.insert(layer * E + (int)(rng() % E));
        std::vector<int> keys(ks.begin(), ks.end());
        // hints for the NEXT call (published before this job, like the predictor one layer ahead)
        next_keys.clear();
        if (rng() % 2) {
            int nl = (layer + 1) % L, nh = (int)(rng() % 8);
            for (int h = 0; h < nh; ++h) {
                int key = nl * E + (int)(rng() % E);
                next_keys.push_back(key);
                __atomic_store_n(hint + 16 + (hint_cnt % HINT_CAP), key, __ATOMIC_RELAXED);
                ++hint_cnt;
            }
            __atomic_thread_fence(__ATOMIC_RELEASE);
            __atomic_store_n(hint, hint_cnt, __ATOMIC_RELEASE);
        }
        if (n == 0) continue;                             // no miss: no job (the wait kernel returns at once)
        ++seq;
        int j = seq % c.jobs;
        int* jb = job(j);
        for (int i = 0; i < n; ++i) jb[4 + i] = keys[i];
        jb[1] = layer; jb[2] = n;
        __atomic_thread_fence(__ATOMIC_RELEASE);
        __atomic_store_n(jb, seq, __ATOMIC_RELEASE);
        // bounded spin (the device wait kernel)
        int64_t t0 = now_ns(), limit = paused ? 2000000LL : 2000000000LL;   // 2 ms while paused, else 2 s
        bool served = false;
        while (now_ns() - t0 < limit) {
            if (__atomic_load_n(done(j), __ATOMIC_ACQUIRE) == seq) { served = true; break; }
        }
        int64_t dt = now_ns() - t0;
        if (served && done(j)[1] == 0) {
            wait_ns_total += dt; wait_ns_max = std::max(wait_ns_max, dt);
            ++ok_jobs;
            const int64_t* lg = reinterpret_cast<const int64_t*>(c.ctl + HLOG_OFF) + (seq % LOG_N) * 4;
            if (lg[0] != seq || lg[1] != n || lg[3] < lg[2] || lg[2] <= 0) ++bad_log;
            for (int i = 0; i < n; ++i) {
                const uint64_t* p = reinterpret_cast<const uint64_t*>((uintptr_t)addr(j)[i]);
                int sl = ckpt_of(keys[i] / E), e = keys[i] % E;
                bool good = true;
                for (int w = 0; w < (int)(REC / 8); w += 97) good &= p[w] == word(sl, e, w);
                good &= p[REC / 8 - 1] == word(sl, e, (int)(REC / 8 - 1));
                bad_pages += !good; ++keys_checked;
            }
        } else {
            ++timeouts;
            if (!paused) ++unexpected_timeouts;
        }
        // release kernel (after the MoE that read the pages)
        __atomic_thread_fence(__ATOMIC_RELEASE);
        __atomic_store_n(rel(j), seq, __ATOMIC_RELEASE);
    }
    svc.pause(false);
    svc.stop();
    printf("jobs %d ok %lld timeouts %lld (unexpected %lld) keys checked %lld bad pages %lld\n", seq, (long long)ok_jobs,
           (long long)timeouts, (long long)unexpected_timeouts, (long long)keys_checked, (long long)bad_pages);
    printf("wait per served job: mean %.1f us, max %.1f us\n", ok_jobs ? wait_ns_total / 1e3 / ok_jobs : 0.0, wait_ns_max / 1e3);
    printf("host: jobs %lld keys %lld demand %lld pf_ready %lld pf_inflight %lld pf_reads %lld hints %lld dropped %lld "
           "known %lld read_err %lld eagain %lld svc_max %.1f us stopped %lld\n",
           (long long)svc.stat(H_JOBS), (long long)svc.stat(H_KEYS), (long long)svc.stat(H_DEMAND_READS),
           (long long)svc.stat(H_PF_HIT_READY), (long long)svc.stat(H_PF_HIT_INFLIGHT), (long long)svc.stat(H_PF_READS),
           (long long)svc.stat(H_HINTS), (long long)svc.stat(H_HINTS_DROPPED), (long long)svc.stat(H_HINTS_KNOWN),
           (long long)svc.stat(H_READ_ERRORS), (long long)svc.stat(H_EAGAIN), svc.stat(H_SVC_NS_MAX) / 1e3,
           (long long)svc.stat(H_STOPPED));
    printf("host job log mismatches %lld\n", (long long)bad_log);
    bool pass = bad_pages == 0 && bad_log == 0 && unexpected_timeouts == 0 && timeouts > 0 && svc.stat(H_PF_HIT_READY) +
                svc.stat(H_PF_HIT_INFLIGHT) > 0 && svc.stat(H_READ_ERRORS) == 0 && svc.stat(H_STOPPED) == 1;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
