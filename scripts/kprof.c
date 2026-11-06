/* kprof.c - in-process kernel+marker trace shim for t2246 seam measurement.
 *
 * Uses the ROCm 10 (rocprofiler 9.0) C API in librocprofiler64.so.2:
 *   - session with an API_TRACE filter over ACTIVITY_DOMAIN_HSA_EVT (kernel
 *     device-activity records with begin/end timestamps) and
 *     ACTIVITY_DOMAIN_ROCTX (markers, bracketed by roctxMarkerStart/End).
 *   - a buffer callback that appends each flushed record to a CSV file:
 *     kind,domain,op_id,thread_id,agent,begin,end,name
 *
 * Link: gcc -shared -fPIC kprof.c -o kprof.so -I<inc> -lrocprofiler64
 * (librocprofiler64.so.2 must be on the ld path; see the box harness).
 */
#include "rocprofiler/v2/rocprofiler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FILE* csv = NULL;
static rocprofiler_session_id_t sess;
static long nrec = 0;

static void dump_record(const rocprofiler_record_header_t* r) {
  if (!csv) return;
  if (r->kind == ROCPROFILER_TRACER_RECORD) {
    const rocprofiler_record_tracer_t* t = (const rocprofiler_record_tracer_t*)r;
    fprintf(csv, "tracer,%d,%u,%u,%u,%llu,%llu,%s\n", (int)t->domain,
            (unsigned)t->operation_id.id, (unsigned)t->thread_id.value,
            (unsigned)t->agent_id.handle, (unsigned long long)t->timestamps.begin.value,
            (unsigned long long)t->timestamps.end.value,
            t->name ? t->name : "");
    nrec++;
  } else {
    fprintf(csv, "other,%d,0,0,0,0,0,\n", (int)r->kind);
  }
}

static void buf_cb(const rocprofiler_record_header_t* begin,
                  const rocprofiler_record_header_t* end,
                  rocprofiler_session_id_t sid, rocprofiler_buffer_id_t bid) {
  const rocprofiler_record_header_t* r = begin;
  int guard = 0;
  while (r && r != end && guard++ < 1000000) {
    dump_record(r);
    const rocprofiler_record_header_t* nx = NULL;
    if (rocprofiler_next_record(r, &nx, sid, bid) != ROCPROFILER_STATUS_SUCCESS) break;
    r = nx;
  }
}

/* init: opens the csv and creates+starts the session. Returns 0 on success. */
int kprof_init(const char* csv_path) {
  csv = fopen(csv_path, "a");
  if (!csv) return 1;
  setvbuf(csv, NULL, _IOLBF, 0);
  rocprofiler_status_t st = rocprofiler_initialize();
  if (st != ROCPROFILER_STATUS_SUCCESS) { fprintf(stderr, "kprof: init status %d\n", st); return 2; }
  rocprofiler_tracer_activity_domain_t doms[2] = {
      (rocprofiler_tracer_activity_domain_t)ACTIVITY_DOMAIN_HSA_EVT,
      (rocprofiler_tracer_activity_domain_t)ACTIVITY_DOMAIN_ROCTX};
  rocprofiler_filter_data_t data;
  data.trace_apis = doms;
  st = rocprofiler_create_ready_session(
      (rocprofiler_replay_mode_t)ROCPROFILER_NONE_REPLAY_MODE, ROCPROFILER_API_TRACE, data,
      2, 256 * 1024 * 1024, buf_cb, &sess, (rocprofiler_filter_property_t){0}, NULL);
  if (st != ROCPROFILER_STATUS_SUCCESS) { fprintf(stderr, "kprof: session status %d\n", st); return 3; }
  st = rocprofiler_start_session(sess);
  if (st != ROCPROFILER_STATUS_SUCCESS) { fprintf(stderr, "kprof: start status %d\n", st); return 4; }
  fprintf(stderr, "kprof: session started\n");
  return 0;
}

/* stop: terminates the session, flushes, finalizes, closes the csv. */
int kprof_stop(void) {
  rocprofiler_terminate_session(sess);
  rocprofiler_flush_data(sess, (rocprofiler_buffer_id_t){0});
  rocprofiler_destroy_session(sess);
  rocprofiler_finalize();
  if (csv) fclose(csv);
  csv = NULL;
  fprintf(stderr, "kprof: stopped, %ld records\n", nrec);
  return 0;
}

/* current HSA timestamp in ns (for host/GPU clock alignment). */
unsigned long long kprof_hsa_now(void) {
  rocprofiler_timestamp_t t;
  if (rocprofiler_get_timestamp(&t) != ROCPROFILER_STATUS_SUCCESS) return 0;
  return t.value;
}
