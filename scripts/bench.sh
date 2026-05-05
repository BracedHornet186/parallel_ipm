#!/usr/bin/env bash
set -u

BIN=./bin
RAW=data/raw_data
MASKS=data/_bench_masks
PTS=data/_bench_points

# Limit frames per run for the benchmark (override with LIMIT=N).
LIMIT="${LIMIT:-100}"

NPROC=$(nproc)
THREADS_LIST=$(echo "1 2 4 8 ${NPROC}" | tr ' ' '\n' | sort -un | tr '\n' ' ')

mkdir -p "$MASKS" "$PTS"

# ---- helpers ----------------------------------------------------------------
need() {
    if [ ! -x "$1" ]; then
        echo "ERROR: missing $1 -- run 'make' first." >&2
        exit 1
    fi
}

# parse the [TIME] line, return seconds (or '?' on failure)
extract_t() {
    awk '/^\[TIME\]/ { print $4; exit }'
}

run_t() {
    # Suppress program stdout, keep stderr (where [TIME] is printed).
    local out
    out=$("$@" 2>&1 >/dev/null)
    local t
    t=$(echo "$out" | extract_t)
    [ -z "$t" ] && t="?"
    echo "$t"
}

speedup() {
    local base="$1" t="$2"
    if [ "$t" = "?" ] || [ "$base" = "?" ]; then echo "  --"; return; fi
    awk -v a="$base" -v b="$t" 'BEGIN { if (b==0) print "  --"; else printf "%4.2fx", a/b }'
}

need "$BIN/lane_detect_serial"
need "$BIN/lane_detect_omp"
need "$BIN/ipm_3d_serial"
need "$BIN/ipm_3d_omp"

# ipm_3d_acc is optional (requires nvc++ at build time).
HAVE_ACC=0
if [ -x "$BIN/ipm_3d_acc" ]; then HAVE_ACC=1; fi

echo "================================================================"
echo "  Benchmark  (limit=${LIMIT} frames per stage)"
echo "  Threads sweep: ${THREADS_LIST}"
echo "================================================================"

# Stage 1 -- lane_detect
echo
echo "[1/2] lane_detect"
echo "  warmup ..."
"$BIN/lane_detect_serial" "$RAW" "$MASKS" --limit 1 >/dev/null 2>&1 || true

S_LANE=$(run_t "$BIN/lane_detect_serial" "$RAW" "$MASKS" --time --limit "$LIMIT")
echo "  serial baseline                    : ${S_LANE} s"

declare -A LP LB LH
for n in $THREADS_LIST; do
    LP[$n]=$(run_t "$BIN/lane_detect_omp" "$RAW" "$MASKS" \
                   --time --limit "$LIMIT" --mode pixel  --threads "$n")
    LB[$n]=$(run_t "$BIN/lane_detect_omp" "$RAW" "$MASKS" \
                   --time --limit "$LIMIT" --mode batch  --threads "$n")
    LH[$n]=$(run_t "$BIN/lane_detect_omp" "$RAW" "$MASKS" \
                   --time --limit "$LIMIT" --mode hybrid --threads "$n")
done

printf "\n  %-9s %9s %8s %9s %8s %9s %8s\n" \
       "threads" "pixel(s)" "speedup" "batch(s)" "speedup" "hybrid(s)" "speedup"
for n in $THREADS_LIST; do
    sp=$(speedup "$S_LANE" "${LP[$n]}")
    sb=$(speedup "$S_LANE" "${LB[$n]}")
    sh=$(speedup "$S_LANE" "${LH[$n]}")
    printf "  %-9s %9s %8s %9s %8s %9s %8s\n" \
           "$n" "${LP[$n]}" "$sp" "${LB[$n]}" "$sb" "${LH[$n]}" "$sh"
done

# Stage 2 -- ipm_3d
echo
echo "[2/2] ipm_3d  (input = $MASKS)"

S_IPM=$(run_t "$BIN/ipm_3d_serial" \
              --input-dir "$MASKS" --output-dir "$PTS" \
              --time --limit "$LIMIT")
echo "  serial baseline                    : ${S_IPM} s"

declare -A IO IA
for n in $THREADS_LIST; do
    IO[$n]=$(run_t "$BIN/ipm_3d_omp" \
                   --input-dir "$MASKS" --output-dir "$PTS" \
                   --time --limit "$LIMIT" --threads "$n")
done

if [ "$HAVE_ACC" -eq 1 ]; then
    echo "  (OpenACC build found -- running GPU sweep too)"
    # warm up GPU once so the first kernel launch isn't counted.
    "$BIN/ipm_3d_acc" --input-dir "$MASKS" --output-dir "$PTS" \
                      --limit 4 --threads 1 >/dev/null 2>&1 || true
    for n in $THREADS_LIST; do
        IA[$n]=$(run_t "$BIN/ipm_3d_acc" \
                       --input-dir "$MASKS" --output-dir "$PTS" \
                       --time --limit "$LIMIT" --threads "$n")
    done
fi

if [ "$HAVE_ACC" -eq 1 ]; then
    printf "\n  %-9s %10s %10s %10s %10s\n" \
           "threads" "omp(s)" "speedup" "acc(s)" "speedup"
    for n in $THREADS_LIST; do
        so=$(speedup "$S_IPM" "${IO[$n]}")
        sa=$(speedup "$S_IPM" "${IA[$n]}")
        printf "  %-9s %10s %10s %10s %10s\n" \
               "$n" "${IO[$n]}" "$so" "${IA[$n]}" "$sa"
    done
else
    printf "\n  %-9s %10s %10s\n" "threads" "omp(s)" "speedup"
    for n in $THREADS_LIST; do
        su=$(speedup "$S_IPM" "${IO[$n]}")
        printf "  %-9s %10s %10s\n" "$n" "${IO[$n]}" "$su"
    done
    echo
    echo "  (build bin/ipm_3d_acc with 'make ipm_3d_acc' to add a GPU column)"
fi

echo
echo "================================================================"
echo "  Done.  Bench artefacts left in: $MASKS, $PTS"
echo "================================================================"
