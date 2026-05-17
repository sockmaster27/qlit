# On the runner machine:
#   docker build -t qlit-bench-runner -f .github/bench-runner/Dockerfile .
#   docker run -d -p 5000:5000 --restart=always --name local-registry registry:2

FROM ubuntu:latest

RUN apt-get update && apt-get install -y mesa-vulkan-drivers build-essential curl
