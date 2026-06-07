#!/bin/bash
set -e
source /usr/bin/gbash.sh || exit 1
DEFINE_string taskset "0c"
DEFINE_string base_dir "experimental/users/sandler/mobilenet/v3_paper_models" ""
DEFINE_string pattern '*.tflite'

gbash::init_google "$@"
BASE_DIR=experimental/users/sandler/mobilenet/v3_paper_models
models=(${BASE_DIR}/${FLAGS_pattern})
echo "${models[@]}"
for model in "${models[@]}" ; do
  cp  -f "${model}" /tmp/
  tmp_model="/tmp/$(basename "${model}")"

  out=$(knowledge/cerebra/brain/compression/mobilenet/tools/tflite_benchmark.sh \
         --model "${tmp_model}" --taskset "${FLAGS_taskset}" \
         --input_layer input  --profile_ops 0 --model_benchmark_binary \
         ~/bin/benchmark_model_236729919 2>&1 || exit 0)

  echo "$(basename "${model}"): $(echo "${out}" | grep -o "no stats:.*") || \
          $(echo "${out}"; exit 1)"
done

