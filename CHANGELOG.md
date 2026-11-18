# CHANGELOG

<!-- version list -->

## v0.1.0-rc.2 (2026-11-18)

### Build System

- Migrate from poetry to uv
  ([`45a0ce8`](https://github.com/sachahu1/diffusion-streaming-api/commit/45a0ce8a05bb9d68e1b667ff627acf4f60cd48f7))

### Chores

- Untrack sam build artifacts
  ([`37e6a0d`](https://github.com/sachahu1/diffusion-streaming-api/commit/37e6a0ddbf1927d72ce49ec24c78d38a0661f0b8))

- **infra**: Drop local aws profile from samconfig
  ([`d645f81`](https://github.com/sachahu1/diffusion-streaming-api/commit/d645f8125dc3efab6fccb458b83a84211d9c10c0))

### Continuous Integration

- Bump test workflow actions to node 24 releases
  ([`0bc1eb1`](https://github.com/sachahu1/diffusion-streaming-api/commit/0bc1eb11e172bbe1304c4990c12770b8997a10c4))

- Call docs workflow directly from release workflow
  ([`86015c7`](https://github.com/sachahu1/diffusion-streaming-api/commit/86015c7ce4a2e0e32526707ad3d62dca0a107e63))

- Push image to ECR and update lambda on release
  ([`28133d3`](https://github.com/sachahu1/diffusion-streaming-api/commit/28133d33e57a0e9224dea8a70e3fe5939beac25e))

### Features

- Build model from checkpoint config and use EMA weights
  ([`d71b93e`](https://github.com/sachahu1/diffusion-streaming-api/commit/d71b93ef4e4df239c540400912625620f5f4d15f))

### Performance Improvements

- Load model once per container and add warm-up endpoint
  ([`48b9bb7`](https://github.com/sachahu1/diffusion-streaming-api/commit/48b9bb7e151ac1f5483ad00a0326f032222f9aa5))


## v0.1.0-rc.1 (2026-11-08)

- Initial Release
