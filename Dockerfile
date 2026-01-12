# Use the CUDA 12.6 base image
FROM 528762/vla-env:latest

# Set working directory to /root
WORKDIR /root

# Install git (if not already installed)
RUN apt-get update && apt-get install -y git

# Clone Bagel_VLA (server branch)
RUN git clone https://github.com/CUHKWilliam/Bagel_VLA.git -b server

# Set the working directory to Bagel_VLA on startup
WORKDIR /root/Bagel_VLA

# Default command (bash shell)
CMD ["/bin/bash"]

