# FROM tensorflow/tensorflow:latest-gpu-jupyter
FROM tensorflow/tensorflow:2.4.2-gpu-jupyter

# ARG USER_ID
# ARG GROUP_ID

# RUN addgroup --gid $GROUP_ID user
# RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
# USER user
ENV JUPYTER_TOKEN="niral"
ENV PASSWORD=niral

RUN apt-get update --fix-missing
RUN apt-get install htop tmux screen -y
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x -o nodesource_setup.sh
RUN bash nodesource_setup.sh && apt-get install -y nodejs

# COPY ../requirements.txt requirements.txt
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip install tensorflow_addons tensorflow_probability==0.12.2 tensorflow_datasets
RUN pip install jupyterlab seaborn pandas numba tqdm pyyaml scipy scikit-learn scipy
RUN pip install scikit-image plotly
RUN pip install -U torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html monai
RUN pip install -U torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 monai

# RUN chmod -R 777 /root
# RUN addgroup --gid 1001 amahmood
# RUN adduser --disabled-password --gecos '' --uid 298493 --gid 1001 amahmood
# USER amahmood
# RUN chmod -R a+wr /root