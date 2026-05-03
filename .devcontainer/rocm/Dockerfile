FROM rocm/dev-ubuntu-24.04:7.2.2-complete
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY --from=ghcr.io/prefix-dev/pixi:latest /usr/local/bin/pixi /usr/local/bin/pixi

RUN userdel -r ubuntu

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
&& useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
&& apt-get update \
&& apt-get install -y sudo git git-lfs \
&& echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
&& chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME