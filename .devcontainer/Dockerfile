FROM dromni/nerfstudio:0.3.4 as base-cuda
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN echo "export LD_LIBRARY_PATH=/home/$USER/local/lib:$LD_LIBRARY_PATH" >> ~/.bashrc \
&& echo "export PATH=/home/$USER/local/bin:$PATH" >> ~/.bashrc \
&& echo "export PATH=/home/$USER/.local/bin:$PATH" >> ~/.bashrc \
&& echo "export PKG_CONFIG_PATH=/home/$USER/local/lib/pkgconfig:$PKG_CONFIG_PATH" >> ~/.bashrc

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics