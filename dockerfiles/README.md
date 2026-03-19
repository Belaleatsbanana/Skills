# Building Docker Images

Some dockerfiles are directly included in this folder and for some others the instructions to build them are below.

The dockerfiles can be built using the standard docker build command. e.g.,
```shell
docker build -t nemo-skills-image:0.7.1 -f dockerfiles/Dockerfile.nemo-skills .
```

In addition, we provide a utility script which provides sane build defaults
```shell
./build.sh Dockerfile.nemo-skills
```

Key configuration environment variables for `build.sh`:
- `DOCKER_NAME`: A fully qualified name of the docker image. The default is inferred from the git repository attributes.
- `DOCKER_TAG`: Docker tag to use. Defaults to `yyyy.mm.dd-<commit_hash>`
- `DOCKER_PUSH`: When set, pushes image after building.
- `DOCKER_PLATFORM`: Directly passed to `--platform` for [multi-platform builds](https://docs.docker.com/build/building/multi-platform/).

## Building for arm64/aarch64

To build for arm64 architecture (e.g. to use with GB200 machines) first follow the installation process at
https://docs.docker.com/build/building/multi-platform/#install-qemu-manually

Then run the same docker command but adding `--platform linux/arm64` or
set `DOCKER_PLATFORM=linux/arm64` for the build script described above.

## Building trtllm image

We directly use official `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc1` image.

## Building sglang image

We directly use official `lmsysorg/sglang:v0.5.8` image.

## Building vllm image

We use official `vllm/vllm-openai:v0.14.1` image with the additional `vllm[audio]` dependencies.

## nemo-rl image

We do not ship a Dockerfile for nemo-rl. Use NVIDIA's pre-built image from NGC with a commit-based tag, e.g. `nvcr.io/nvidian/nemo-rl:9148186-44694499`. Set this in your cluster config under `containers.nemo-rl` (see `cluster_configs/example-local.yaml`). Replace the tag with the desired commit/build id.
