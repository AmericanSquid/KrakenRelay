# UVK5D PTT Plugin

This plugin adds UV-K5/uvk5d direct PTT support to KrakenRelay without directly editing core PTT files.

KrakenRelay's current plugin API does not expose a formal PTT backend registry yet, so this plugin installs a small runtime patch when loaded. That lets `PTTManager` build and operate a `UVK5D` backend while keeping the core tree clean.

## Install

From the KrakenRelay repo root:

```bash
tar -xzf krakenrelay-plugin-uvk5d-ptt.tar.gz -C plugins
python3 plugins/install.py
```

Choose `uvk5d_ptt` if prompted.

Direct hook install also works:

```bash
python3 plugins/uvk5d_ptt/install.py
```

## Plugin config

`plugins/config.yaml` should contain:

```yaml
enabled:
  - uvk5d_ptt
```

The installer hook will add this automatically if it can find the repo root.

## Main PTT config

Single PTT:

```yaml
ptt:
  dual_ptt: false
  mode: UVK5D
  uvk5d_host: 127.0.0.1
  uvk5d_port: 7355
  uvk5d_timeout: 0.5
```

Dual PTT example:

```yaml
ptt:
  dual_ptt: true
  primary:
    mode: CM108
    device_path: /dev/hidraw0
    gpio_pin: 3
  secondary:
    mode: UVK5D
    uvk5d_host: 127.0.0.1
    uvk5d_port: 7355
    uvk5d_timeout: 0.5
```

## Runtime behavior

When loaded, the plugin:

- treats `UVK5D`, `UV-K5D`, and `UV_K5D` as the same mode
- patches `PTTManager._build_ptt()` so it can construct `UVK5DPTT`
- patches hardware PTT detection so UVK5D is treated like a hardware PTT backend
- preserves the existing safe key/unkey fallback behavior
- patches TX control so UVK5D hardware PTT does not trigger the VOX carrier-delay wake burst
- exposes simple plugin status through `status()` / `api_status()`

## Uninstall

```bash
python3 plugins/uninstall.py
```

or:

```bash
python3 plugins/uvk5d_ptt/uninstall.py
```

Then restart KrakenRelay.

