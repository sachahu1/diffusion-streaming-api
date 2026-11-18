# diffusion-streaming-api


## Installation
### Installing uv
This tool uses [uv](https://docs.astral.sh/uv/). If you already have uv installed,
please skip to the next section. Otherwise, let's first setup uv.

To install uv, simply run this command:
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```
You can find out more about uv installation [here](https://docs.astral.sh/uv/getting-started/installation/).

That's it, uv is set up.

### Installing the package
Thanks to uv, installing this package is very simple and can be done in a single command. Simply run:
```shell
uv sync
```
That's it, the package is installed. Move to the next section to learn how to use this package.

## Getting Started
< Add instructions on how to use project here >
## Building the documentation
To build the documentation you can simply use the docker image. To do so, simply run:
```shell
docker build . -f Dockerfile --target documentation -t diffusion_streaming_api-docs
```
