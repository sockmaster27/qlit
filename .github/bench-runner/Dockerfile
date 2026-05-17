# To allow the benchrmarking workflows to pick up this image, run this on the runner machine:
#   docker run -d -p 5000:5000 --restart=always --name local-registry registry:2
#   docker build -t localhost:5000/qlit-bench-runner -f .github/bench-runner/Dockerfile .
#   docker push localhost:5000/qlit-bench-runner

FROM ubuntu:latest

RUN apt-get update && apt-get install -y mesa-vulkan-drivers build-essential curl
