# primitive_fitting

Scaffolded by `xInsp2/sdk/scaffold.mjs`. This README is your offline kit
— everything you need to build, test, and ship a plugin without leaving
this folder.

---

## Contents

1. [Build & load](#build--load)
2. [Tests](#tests)
3. [The plugin interface](#the-plugin-interface)
4. [`xi::Record` — the universal data bag](#xirecord--the-universal-data-bag)
5. [`xi::Json` — RAII cJSON wrapper](#xijson--raii-cjson-wrapper)
6. [`xi::Image`](#xiimage)
7. [Per-instance folder storage](#per-instance-folder-storage)
8. [Webview UI protocol](#webview-ui-protocol)
9. [`plugin.json` manifest](#pluginjson-manifest)
10. [Lifecycle](#lifecycle)
11. [Common patterns](#common-patterns)
12. [Pitfalls](#pitfalls)
13. [Layout](#layout)
14. [Where to look next](#where-to-look-next)

---

## Build & load

```bat
:: First time only — locate xInsp2 (auto-detected if it's a sibling)
set XINSP2_ROOT=C:\path\to\xInsp2

cmake -S . -B build -A x64
cmake --build build --config Release
```

Output: `primitive_fitting.dll` next to `plugin.json`. The host expects this layout.

Pick one to load it:

- **VS Code**: Settings → `xinsp2.extraPluginDirs` → add the **parent** folder
- **CLI**: `xinsp-backend.exe --plugins-dir=<parent-folder>`
- **Env var**: `set XINSP2_EXTRA_PLUGIN_DIRS=<parent-folder>`

In VS Code: click `+` in **Instances** view → pick `primitive_fitting` → name it.
The webview opens via the gear icon or single-click on the tree row.

The host runs an 8-test **baseline** (C-ABI safety) on first load and
writes `cert.json` next to the DLL. Subsequent loads with the same DLL
skip the baseline. Failure quarantines the DLL.

---

## Tests

### Native (C++) — `tests/test_native.cpp`

Loads the DLL via `LoadLibrary`, runs baseline + your own `XI_TEST(...)`
blocks. Builds to `primitive_fitting_test.exe`.

```bat
cmake --build build --config Release --target primitive_fitting_test
.\primitive_fitting_test.exe
```

Or via ctest from the build dir:

```bat
cd build && ctest -C Release --output-on-failure
```

A passing run writes `cert.json`, so the next host load skips baseline.
Add tests with:

```cpp
XI_TEST(my_specific_thing) {
    load_dll();
    void* inst = g_syms.create(&g_host, "t");
    // … exercise via g_syms.process / g_syms.exchange …
    XI_EXPECT(some_invariant);
    XI_EXPECT_EQ(actual, expected);
    g_syms.destroy(inst);
}
```

### UI E2E (JavaScript) — `tests/test_ui.cjs`

Drives the live VS Code + backend stack with a real webview. Two ways:

**Cold session (CI-friendly):**
```bat
node %XINSP2_ROOT%\sdk\testing\run_ui_test.mjs .
```

**Warm session (faster inner loop):**
```
Cmd Palette → "xInsp2: Run Plugin UI Tests" → pick this folder
```

Same `test_ui.cjs`, same `h` helpers in both. Screenshots → `tests/screenshots/`.

`h` cheatsheet:

| Method | Purpose |
|--------|---------|
| `h.createProject(folder, name)` | run `xinsp2.createProject` |
| `h.addInstance(name, plugin)`   | run `xinsp2.createInstance` |
| `h.openUI(name, plugin)`        | open the webview, wait for mount |
| `h.click(inst, selector)`       | dispatch a real click in the webview |
| `h.setInput(inst, sel, value)`  | set value, fire `input` + `change` events |
| `h.sendCmd(inst, cmd)`          | exchange round-trip; updates `h.lastStatus` |
| `h.getStatus(inst)`             | shorthand for `sendCmd({command:'get_status'})` |
| `h.run()`                       | run `xinsp2.run` (one inspection cycle) |
| `h.shot(label)`                 | full-screen PNG into `tests/screenshots/` |
| `h.expect(cond, msg)`           | soft assertion |
| `h.expectEq(a, b, msg)`         | structural equality |
| `h.sleep(ms)`, `h.tmp()`        | utilities |
| `h.lastStatus`                  | last successful exchange's parsed payload |

---

## The plugin interface

Inherit `xi::Plugin`, override what you need, end with `XI_PLUGIN_IMPL(YourClass)`:

```cpp
#include <xi/xi_abi.hpp>

class PrimitiveFitting : public xi::Plugin {
public:
    using xi::Plugin::Plugin;

    xi::Record process(const xi::Record& input) override;
    std::string exchange(const std::string& cmd) override;
    std::string get_def() const override;
    bool        set_def(const std::string& json) override;
};

XI_PLUGIN_IMPL(PrimitiveFitting)
```

Defaults (override only what you need):

| Method | Default | When to override |
|--------|---------|------------------|
| `xi::Record process(const xi::Record&)` | returns `{}` | every plugin: this is the per-frame hot path |
| `std::string exchange(const std::string&)` | returns `"{}"` | when the UI sends commands |
| `std::string get_def() const` | returns `"{}"` | to persist config across host restarts |
| `bool set_def(const std::string&)` | returns `true` | to restore that config |
| `void start()` / `void stop()` | no-op | streaming sources (cameras) only |

`xi::ImageSource` is a subclass for cameras — adds `grab()` /
`grab_wait()` and a built-in frame queue with backpressure. For
correlated multi-camera capture, use `host()->emit_trigger(...)`
directly — see the SDK README's *Image sources and the trigger bus*
section and the `trigger_source/` example.

`name()` returns this instance's name. `host()` returns the host API.

---

## `xi::Record` — the universal data bag

A record bundles **named images + JSON data** in one object — the
input/output type for `process()`, the way scripts pipe data between
plugins. Backed by cJSON; thread-safe to read; copy is shallow for
image bytes (shared_ptr-backed).

```cpp
// Build
xi::Record r;
r.set("count", 5)
 .set("pass", true)
 .set("label", "ok")
 .image("binary", img)
 .image("overlay", rgb);

// Nested objects
r.set("roi", xi::Record().set("x", 10).set("y", 20));

// Read with defaults — never crash on missing/wrong-type
int  n   = r["count"].as_int(0);
bool ok  = r["pass"].as_bool(false);
auto lbl = r["label"].as_string("");

// Path access
int x     = r["roi.x"].as_int();
int first = r["points[0].value"].as_int();

// Images
const xi::Image& img = r.get_image("binary");
for (auto& [key, img] : r.images()) { /* iterate */ }
```

---

## `xi::Json` — RAII cJSON wrapper

For parsing `exchange()` commands and building reply payloads. No manual
`cJSON_Delete`. Same path syntax as `xi::Record`.

```cpp
#include <xi/xi_json.hpp>

// Parse
auto p = xi::Json::parse(cmd);
std::string command = p["command"].as_string();   // "" if missing
int n     = p["value"].as_int(0);                 // default if missing/wrong type
double t  = p["roi.threshold"].as_double(128.0);  // path access

// Iterate
p["points"].for_each([&](const char* idx, xi::Json v) {
    int x = v["x"].as_int();
});

// Build
auto reply = xi::Json::object()
    .set("ok", true)
    .set("count", 42)
    .set("nested", xi::Json::object().set("k", "v"));

auto arr = xi::Json::array().push(1).push(2).push(3);
reply.set("nums", arr);

return reply.dump();         // compact
// or reply.dump_pretty();   // indented
```

A typical exchange handler shrinks from ~12 lines of cJSON_*/null-checks
to 3.

---

## `xi::Image`

```cpp
xi::Image img(width, height, channels);   // allocates
uint8_t* p = img.data();                   // raw access
int n = img.width * img.height * img.channels;

// Read input image
const xi::Image& src = input.get_image("src");
if (src.empty()) return xi::Record().set("error", "no src");

// Make output, write into output Record
xi::Image out(src.width, src.height, src.channels);
// … fill out.data() …
return xi::Record().image("dst", out);
```

`xi::Image` holds a `shared_ptr` to pixels — copies are cheap. Pixel
storage is owned by the host's `ImagePool`, refcounted. Don't free
manually.

---

## Per-instance folder storage

Every instance gets a dedicated on-disk folder for permanent data
larger than the small JSON config (calibration files, reference images,
LUTs, ML weights — anything you'd lose with `get_def`/`set_def`).

```cpp
std::string folder = folder_path();
auto path = std::filesystem::path(folder) / "ref.png";
std::ofstream f(path.string(), std::ios::binary);
// … write bytes …
```

The path is `<project>/instances/<instance_name>/`. Properties:

- **Per instance, not per plugin.** Two instances of the same plugin
  each get their own folder
- **Created before your constructor runs**, so it's safe to write from
  `xi_plugin_create()` time
- **Never deleted by the host** — survives hot-reload, project
  open/close, host restart, instance recreate. Only the user can
  delete it
- **Inside the project folder** — copying or zipping the project
  carries all your instance data along
- `instance.json` (the host's serialization of `get_def()`) lives in
  the same folder

Returns empty string if the plugin is running detached from a project.

---

## Webview UI protocol

`ui/index.html` is a plain HTML+JS file. The bridge between it and your
C++ plugin is `vscode.postMessage` — but the round-trip is asymmetric
in a way worth understanding before you write your first UI.

### The round-trip, end to end

```
   ┌────────────────────────────────────────────────────────────┐
   │                       webview (HTML+JS)                    │
   │                                                            │
   │   button.onclick                                           │
   │     └─ vscode.postMessage(                                 │
   │           { type:'exchange',                               │
   │             cmd: { command:'set_x', value:42 } });         │
   │                       │                                    │
   └───────────────────────┼────────────────────────────────────┘
                           │ webview → extension
                           ▼
   ┌────────────────────────────────────────────────────────────┐
   │                    VS Code extension                       │
   │                                                            │
   │   panel.webview.onDidReceiveMessage(msg => {               │
   │     sendCmd('exchange_instance',                           │
   │             { name:'inst0', cmd: msg.cmd })                │
   │       .then(rsp => panel.webview.postMessage(              │
   │              { type:'status', ...JSON.parse(rsp.data) }))  │
   │   });                                                      │
   └───────────────────────┬────────────────────────────────────┘
                           │ extension → backend (WebSocket)
                           ▼
   ┌────────────────────────────────────────────────────────────┐
   │                     backend + plugin                       │
   │                                                            │
   │   plugin.exchange(cmd_json)  ← YOUR C++                    │
   │     ├─ parse with xi::Json                                 │
   │     ├─ mutate state                                        │
   │     └─ return get_def();   // emits FULL state as JSON     │
   └───────────────────────┬────────────────────────────────────┘
                           │ rsp.data = your returned string
                           ▼ extension parses + posts back
   ┌────────────────────────────────────────────────────────────┐
   │                       webview (HTML+JS)                    │
   │                                                            │
   │   window.addEventListener('message', e => {                │
   │     if (e.data.type === 'status')                          │
   │       updateUI(e.data);   // re-render from full state     │
   │   });                                                      │
   └────────────────────────────────────────────────────────────┘
```

### Three things to internalize

**1. The plugin doesn't push to the UI — it returns a string.**
Your `exchange()` returns JSON; the extension wraps it as
`{type:'status', ...parsed_json}` and posts to the webview. You never
need to know the webview exists.

**2. Always return `get_def()` from `exchange()`.**
That way the UI gets the full state on every reply, regardless of
which command was issued. Your UI code becomes a single
"render-from-state" function — no diffing, no per-command response
shapes.

**3. The first render is just another `exchange()` call.**
On mount, the UI sends `{command:'get_status'}`. Your C++ doesn't need
to recognize that name — unknown commands fall through to `get_def()`,
which is exactly what the UI wants. One handler, two paths.

### Other message types the extension understands

| Message from webview | Effect |
|---------|--------|
| `{type:'exchange', cmd: {…}}` | round-trip to your `exchange()`, status posted back |
| `{type:'request_preview'}`    | plays nicely with `xi::ImageSource` previews |
| `{type:'request_process', cmd:{…}}` | result posted as `{type:'process_result', …}` |

### Writing it: minimal pattern

UI side:

```js
const vscode = acquireVsCodeApi();
function send(cmd) { vscode.postMessage({ type: 'exchange', cmd }); }

// User actions → send()
function apply() { send({ command: 'set_threshold', value: 128 }); }

// Inbound state → DOM
window.addEventListener('message', e => {
    const m = e.data;
    if (m.type === 'status' && 'threshold' in m) {
        document.getElementById('threshVal').textContent = m.threshold;
    }
});

// Initial state pull
send({ command: 'get_status' });
```

C++ side:

```cpp
std::string exchange(const std::string& cmd) override {
    auto p = xi::Json::parse(cmd);
    auto command = p["command"].as_string();
    if      (command == "set_threshold") threshold_ = p["value"].as_int(threshold_);
    else if (command == "set_invert")    invert_    = p["value"].as_bool(invert_);
    return get_def();   // unknown commands also land here — safe default
}

std::string get_def() const override {
    return xi::Json::object()
        .set("threshold", threshold_)
        .set("invert",    invert_)
        .dump();
}
```

That's the whole protocol. Look at `primitive_fitting.cpp` and `ui/index.html`
in this folder — they're a working implementation.

---

## `plugin.json` manifest

```json
{
  "name": "primitive_fitting",
  "description": "What it does (shown in the + picker)",
  "dll": "primitive_fitting.dll",
  "factory": "xi_plugin_create",
  "has_ui": true
}
```

- `name` must be unique across all plugins on disk
- `dll` is relative to the plugin folder
- `has_ui: true` → host expects `ui/index.html`
- `factory` should always be `xi_plugin_create` (provided by `XI_PLUGIN_IMPL`)

---

## Lifecycle

```
host scans plugin.json
         ↓
host runs baseline (8 tests) on first load, writes cert.json on pass
         ↓
host calls xi_plugin_create(host_api, name)            → new PrimitiveFitting(...)
         ↓
host calls xi_plugin_set_def(stored_config_json)       → restore config
         ↓
┌──── for each inspection frame ──────────────────────────────────┐
│ host calls xi_plugin_process(input_record, out_record)          │
│ host calls xi_plugin_exchange(cmd_json, reply_buf) on UI clicks │
└─────────────────────────────────────────────────────────────────┘
         ↓ (project save)
host calls xi_plugin_get_def() → persisted to instance.json
         ↓ (shutdown / unload)
host calls xi_plugin_destroy(inst)
```

`XI_PLUGIN_IMPL(PrimitiveFitting)` generates all six C-ABI exports for you.

---

## Common patterns

### A simple processor (image in → image out)

```cpp
xi::Record process(const xi::Record& input) override {
    const xi::Image& src = input.get_image("src");
    if (src.empty()) return xi::Record().set("error", "no 'src' image");

    xi::Image dst(src.width, src.height, src.channels);
    const uint8_t* sp = src.data();
    uint8_t*       dp = dst.data();
    const int n = src.width * src.height * src.channels;
    for (int i = 0; i < n; ++i) dp[i] = (uint8_t)(255 - sp[i]);

    return xi::Record().image("dst", dst);
}
```

### Saving a binary file in the instance folder

```cpp
auto folder = std::filesystem::path(folder_path());
std::filesystem::create_directories(folder);
std::ofstream f((folder / "lut.bin").string(), std::ios::binary);
f.write((const char*)lut, 256);
```

### Persisting state via get/set_def

```cpp
std::string get_def() const override {
    return xi::Json::object()
        .set("threshold", threshold_)
        .set("invert",    invert_)
        .dump();
}

bool set_def(const std::string& j) override {
    auto p = xi::Json::parse(j);
    if (!p.valid()) return false;
    threshold_ = p["threshold"].as_int(threshold_);
    invert_    = p["invert"].as_bool(invert_);
    return true;
}
```

### Routing UI commands

```cpp
std::string exchange(const std::string& cmd) override {
    auto p = xi::Json::parse(cmd);
    auto command = p["command"].as_string();

    if      (command == "set_threshold") threshold_ = p["value"].as_int(threshold_);
    else if (command == "set_invert")    invert_    = p["value"].as_bool(invert_);
    else if (command == "reset")         { threshold_ = 128; invert_ = false; }
    // 'get_status' falls through; we just return current state.

    return get_def();
}
```

### Allowing per-frame overrides via input

```cpp
xi::Record process(const xi::Record& input) override {
    int t = input["threshold"].as_int(threshold_);   // input wins, else stored
    // …use t…
}
```

Lets a script do `inst.process(xi::Record().image("src", img).set("threshold", 200))`
to try a value for one frame without touching the persistent config.

---

## Pitfalls

- **`process()` runs on the inspection thread.** Don't block on
  network/hardware — spawn a worker or subclass `xi::ImageSource`.
- **`get_image()` returns a const reference** — check `.empty()` before
  using.
- **The `xi_host_api` pointer your plugin holds is per-instance** —
  don't share it between unrelated objects.
- **Hot reload**: instance state survives via `get_def`/`set_def`. If
  you cache mutable state outside that pair (a static, a singleton), it
  won't survive — move it into your class.
- **Cert is invalidated by DLL changes**: rebuild → next load re-runs
  baseline. Failing baseline blocks instantiation. Run
  `primitive_fitting_test.exe` locally to write cert proactively.
- **Webview `exchange` posts go through the extension wrapper** — the
  reply comes back as `{type:'status', ...}`. Don't expect a direct
  return value.

---

## Layout

```
primitive_fitting/
├── plugin.json            ← manifest
├── primitive_fitting.cpp          ← all your C++ goes here (or split as you grow)
├── ui/index.html          ← webview UI
├── CMakeLists.txt         ← uses xInsp2's cmake helper
├── tests/
│   ├── test_native.cpp    ← XI_TEST blocks; compiles to primitive_fitting_test.exe
│   ├── test_ui.cjs        ← exports run(h); drives the live UI
│   └── screenshots/       ← created by h.shot(...)
├── primitive_fitting.dll          ← build output (placed beside plugin.json)
├── primitive_fitting_test.exe     ← native test runner
├── cert.json              ← written by host (or your test) on baseline pass
├── lab/                   ← algorithm research (see lab/RESULTS.md)
│   └── cnn/               ← CNN training pipeline
└── build/                 ← cmake out-of-tree dir
```

---

## CNN edge-filter mode (optional)

The plugin's hand-crafted gradient + NMS peak extractor can be
replaced by a learned 1-D / 2-D CNN with two config flags. Either
the per-caliper [N, 15, 80] CaliperEdgeNet or the cross-caliper
[1, N, 15, 80] CrossCaliperEdgeNet works — input rank is auto-
detected when the ONNX file is loaded.

Enable via UI checkbox or JSON config:
```json
{ "use_cnn_peak_filter": true,
  "cnn_onnx_path":       "C:/path/to/caliper_edge_v5lite.onnx" }
```

Constraints (must match training):
* `caliper_width` is effectively forced to 3 for v3 models, 15 for v4/v5
* `caliper_span`  is effectively forced to 80
* The model file is loaded once and cached; reload triggers on path change.

Failure handling: if the ONNX is missing, mismatched-shape, or the
forward call throws, the plugin returns no hits for the image (rather
than silently swapping in the classical extractor). This makes
quality regressions obvious instead of disguised.

### Training pipeline

The full CNN training pipeline lives in `lab/cnn/`:

```sh
# 1. Generate dataset (5000 scenes × 16 calipers, scene-grouped):
./lab/build/Release/dump_caliper_dataset.exe \
    lab/cnn/data/normal_sc.bin --scenes 5000 --scene-records
./lab/build/Release/dump_caliper_dataset.exe \
    lab/cnn/data/harsh_sc.bin  --scenes 3000 --harsh --scene-records
./lab/build/Release/dump_caliper_dataset.exe \
    lab/cnn/data/photo_sc.bin  --scenes 5000 --photo --scene-records

# 2. Train CrossCaliperEdgeNet (≈10 min on CPU):
cd lab/cnn
python -m venv .venv
.venv/Scripts/pip install "torch==2.5.1" \
    --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install numpy onnx matplotlib
.venv/Scripts/python.exe train.py \
    data/normal_sc.bin data/harsh_sc.bin data/photo_sc.bin \
    --epochs 60 --bs 32 --patience 10 --out caliper_edge_v5.pt

# Optional: monitor every 5 min
.venv/Scripts/python.exe plot_metrics.py caliper_edge_v5.pt.metrics.csv

# 3. Export to ONNX:
.venv/Scripts/python.exe export_onnx.py \
    caliper_edge_v5.pt --out caliper_edge_v5.onnx --K 16

# 4. Point the plugin at it:
#    UI → set "ONNX path" to lab/cnn/caliper_edge_v5.onnx and tick the box.
```

Lab-bench (100 random scenes per regime, see `lab/RESULTS.md` for
the full breakdown):

| Algorithm                   | Outlier (n / h / p) | RMS p50 (n / h / p) | ms |
|-----------------------------|--------------------:|--------------------:|---:|
| `caliper_cnn_prosac`        | 0% / 0% / 0%        | 0.17 / 0.18 / 0.16  | 0.65 |
| `caliper_cnn_ort` (v5-lite) | 0% / 0% / 0%        | 0.20 / 0.23 / 0.19  | 0.78 |
| `caliper_ransac` (default)  | 4% / 74% / 7%       | 0.20 / 0.04* / 0.22 | 0.43 |

*caliper_ransac p50 ≈ 0 in harsh is a fail-statistic artefact.

In practice: a model trained with `lab/cnn/train.py` plus
`fit_model = "polynomial"` in plugin config gives the same DP+poly
+ safety-net behaviour as the lab's `caliper_cnn_prosac` — the CNN
emits clean per-caliper hits, the plugin's adaptive-degree poly
RANSAC fits a smooth curve through them, IRLS refines, the
confidence pipeline reports stability/coverage, etc.

Project-scoped per-instance data lives elsewhere — under
`<project>/instances/<instance_name>/` — accessible via
`xi::Plugin::folder_path()`.

---

## Where to look next

- **Worked examples** — `%XINSP2_ROOT%\sdk\examples\`
  - `hello/`     — minimal: 15 lines, no state, no UI
  - `counter/`   — state + persistence + UI + tests
  - `invert/`    — image in → image out
  - `histogram/` — image analysis with rich JSON output + canvas UI
- **Real-world plugin** — `<plugins>/ct_shape_based_matching/`
  (OpenCV + AVX2 + UI + per-instance template storage)
- **Full ABI definition** — `%XINSP2_ROOT%\backend\include\xi\xi_abi.h`
- **Test framework + baseline** — `%XINSP2_ROOT%\backend\include\xi\xi_test.hpp`,
  `xi_baseline.hpp`
- **UI helpers source** — `%XINSP2_ROOT%\sdk\testing\helpers.cjs`

---

## Tips

- **Hot reload**: rebuild the DLL, the host reloads it, instance state
  survives via `get_def`/`set_def`
- **Debug prints**: `std::fprintf(stderr, "[primitive_fitting] %d\n", x)` shows
  up in the backend's output channel
- **Don't modify `xInsp2/`** — `git pull` on the host should never
  touch your work
- **Class name + file name**: scaffold emits `PrimitiveFitting` (Pascal) +
  `primitive_fitting` (snake) + `PRIMITIVE_FITTING` (UPPER for macros) — keep them in
  sync if you rename
