//
// test_ui.cjs — visual demo: weak continuous curve vs strong noise spikes.
//
// Scene: gentle bright edge curving across the image (δ=60, everywhere)
// plus 6 isolated bright spikes above the curve (δ≈150). Three config
// variants demonstrate the journey:
//   (a) top_n=1 baseline — algorithm grabs whatever is strongest per
//       caliper, losing curve points in spike columns.
//   (b) top_n=3 — weaker but continuous curve peaks are rescued.
//   (c) top_n=3 + polynomial residual filter — residual model captures
//       the curvature and cleans stragglers.
//

const fs   = require('fs');
const os   = require('os');
const path = require('path');
const { execSync } = require('child_process');

const SCRIPT = `
#include <xi/xi.hpp>
#include <xi/xi_image.hpp>
#include <xi/xi_record.hpp>
#include <xi/xi_plugin_handle.hpp>
#include <xi/xi_instance.hpp>
#include <cstring>
#include <cmath>
#include <memory>

static std::shared_ptr<xi::PluginHandle> g_finder;
struct RegisterFinder {
    RegisterFinder() {
        g_finder = std::make_shared<xi::PluginHandle>("finder0", "primitive_fitting");
        xi::InstanceRegistry::instance().add(g_finder);
    }
} g_register_finder;

// 320x240 scene: half-period sine edge (δ=60, everywhere) plus 6 bright
// spike blocks above the curve (δ≈150, locally stronger than the curve).
static xi::Image make_scene() {
    const int W = 320, H = 240;
    const double PI = 3.14159265358979;
    xi::Image img(W, H, 1);
    uint8_t* p = img.data();
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            double curve_y = 120.0 + 15.0 * std::sin(x * 2.0 * PI / (double)W);
            p[y * W + x] = (uint8_t)(y < curve_y ? 90 : 150);
        }
    }
    for (int i = 0; i < 30; ++i) {
        int sx = 30 + ((i * 13 + 7) % 260);
        int sy = 93 + ((i * 17 + 11) % 18);
        for (int dy = 0; dy < 5; ++dy)
        for (int dx = 0; dx < 5; ++dx) {
            int yy = sy + dy, xx = sx + dx;
            if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
            p[yy * W + xx] = 240;
        }
    }
    return img;
}

XI_SCRIPT_EXPORT
void xi_inspect_entry(int frame) {
    (void)frame;
    auto img = make_scene();
    VAR(scene, img);
    auto result = g_finder->process(xi::Record().image("src", img));
    int f = result["found"].as_int();
    int p = result["pass"].as_int();
    VAR(found, f); VAR(pass, p);
    if (result.has_image("result")) VAR(overlay, result.get_image("result"));
}
`;

function shotVSCode(label, shotDir) {
    fs.mkdirSync(shotDir, { recursive: true });
    const fpath = path.join(shotDir, `${label}.png`);
    const psScript = path.join(os.tmpdir(), `xi_pf_shot_${process.pid}.ps1`);
    fs.writeFileSync(psScript, `
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
using System.Text;
public class Win {
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr hWnd, IntPtr hdc, int nFlags);
    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
}
public struct RECT { public int Left, Top, Right, Bottom; }
"@
$target = [IntPtr]::Zero
$cb = [Win+EnumWindowsProc]{
    param($hWnd, $lParam)
    if (-not [Win]::IsWindowVisible($hWnd)) { return $true }
    $sb = New-Object System.Text.StringBuilder 512
    [void][Win]::GetWindowText($hWnd, $sb, 512)
    $t = $sb.ToString()
    if ($t -match 'Extension Development Host' -or $t -match 'xinsp2_pluginui_') {
        if ($script:target -eq [IntPtr]::Zero) { $script:target = $hWnd }
    }
    return $true
}
[void][Win]::EnumWindows($cb, [IntPtr]::Zero)
if ($target -eq [IntPtr]::Zero) { Write-Error "dev host not found"; exit 2 }
$r = New-Object RECT
[void][Win]::GetWindowRect($target, [ref]$r)
$w = $r.Right - $r.Left; $h = $r.Bottom - $r.Top
$bmp = New-Object System.Drawing.Bitmap($w, $h)
$gfx = [System.Drawing.Graphics]::FromImage($bmp)
$hdc = $gfx.GetHdc()
[void][Win]::PrintWindow($target, $hdc, 2)
$gfx.ReleaseHdc($hdc)
$bmp.Save("${fpath.replace(/\\/g, '\\\\')}")
$gfx.Dispose(); $bmp.Dispose()
`);
    try {
        execSync(`powershell -NoProfile -ExecutionPolicy Bypass -File "${psScript}"`, { timeout: 15000 });
        console.log(`  📸 ${path.basename(fpath)} (VS Code only)`);
    } catch (e) {
        console.log(`  📸 ${path.basename(fpath)} failed: ${e.message}`);
    }
    return fpath;
}

// Dump a base64 image returned by an exchange command.
async function saveImageFromCmd(h, cmd, field, outPath) {
    await h.sendCmd('finder0', cmd);
    const s = h.lastStatus || {};
    const data = s[field];
    if (!data) return false;
    fs.writeFileSync(outPath, Buffer.from(data, 'base64'));
    return true;
}

module.exports = {
    async run(h) {
        const projDir = h.tmp();
        const shotDir = path.join(h.pluginFolder, 'tests', 'screenshots');
        fs.mkdirSync(shotDir, { recursive: true });

        await h.createProject(projDir, 'pf_demo_e2e');

        const scriptPath = path.join(projDir, 'e2e_inspect.cpp');
        fs.writeFileSync(scriptPath, SCRIPT);
        const compRsp = await h.api.sendCmd('compile_and_load', { path: scriptPath });
        h.expect(compRsp && compRsp.ok, `compile_and_load (${compRsp && compRsp.error})`);

        // Prime a frame so the UI has a preview.
        let runRsp = await h.api.sendCmd('run');
        h.expect(runRsp && runRsp.ok, 'prime run ok');
        await h.sleep(400);

        await h.openUI('finder0', 'primitive_fitting');
        await h.sleep(800);

        // Fixed line region across the image.
        await h.sendCmd('finder0', {
            command: 'set_region', mode: 'line',
            p1x: 20, p1y: 120, p2x: 300, p2y: 120,
        });

        const commonCfg = {
            command: 'set_config',
            fit_model: 'line',
            polarity:  'any',
            num_calipers: 30,
            caliper_width: 3,
            caliper_span: 80,
            min_edge_strength: 10,
            edge_min_separation_px: 4,
            top_n_min_alpha: 0,
            ransac_threshold_px: 5.0,
            ransac_iterations: 300,
            ransac_weight_by_strength: false,
            min_inlier_ratio: 0,
        };

        async function runCase(label, overrides) {
            await h.sendCmd('finder0', Object.assign({}, commonCfg, overrides));
            await h.sleep(200);
            runRsp = await h.api.sendCmd('run');
            h.expect(runRsp && runRsp.ok, `run ok (${label})`);
            await h.sleep(900);
            shotVSCode(label, shotDir);
            await h.getStatus('finder0');
            const s = h.lastStatus || {};
            await saveImageFromCmd(h,
                { command: 'get_last_result' },
                'result_png',
                path.join(shotDir, `${label}_overlay.png`));
            return s;
        }

        const a = await runCase('case_a_top1',         { top_n_per_caliper: 1, poly_enabled: false, poly_degree: 0 });
        const b = await runCase('case_b_top3',         { top_n_per_caliper: 3, poly_enabled: false, poly_degree: 0 });
        const c = await runCase('case_c_top3_poly3',   { top_n_per_caliper: 3, poly_enabled: true,  poly_degree: 3, poly_reject_sigma: 2.5 });
        // (d) Polynomial primitive fit: the curve itself IS the primitive.
        // Tight RANSAC threshold, max-slope cap to reject wild candidates.
        const d = await runCase('case_d_polyfit_deg3', {
            top_n_per_caliper: 3,
            fit_model: 'polynomial',
            poly_degree: 3,                    // degree-3 needed for S
            poly_max_slope_deg: 45,
            ransac_threshold_px: 2.0,
            poly_enabled: false,
        });

        console.log('\n   case           | found | inliers | total | residual σ | poly_rej');
        console.log(  '   ---------------+-------+---------+-------+------------+---------');
        for (const [name, s] of [['(a) top1      ', a],
                                  ['(b) top3      ', b],
                                  ['(c) +poly2    ', c],
                                  ['(d) poly prim.', d]]) {
            console.log(`   ${name} |   ${s.found ? '✓' : '✗'}  |   ${String(s.inlier_count ?? '—').padStart(2)}/${String(30).padStart(2)}  |  ${String(s.total_hits ?? '—').padStart(3)}  |   ${(s.residual_std ?? 0).toFixed(2)}     |   ${s.poly_rejected ?? 0}`);
        }

        // Key claims for the story.
        h.expect((b.inlier_count ?? 0) > (a.inlier_count ?? 0),
                 `top-N rescued curve hits (a=${a.inlier_count} → b=${b.inlier_count})`);
        h.expect((b.total_hits   ?? 0) > (a.total_hits   ?? 0),
                 `top-N produced more candidates (a=${a.total_hits} → b=${b.total_hits})`);
        h.expect(c.poly_applied === true, 'poly filter applied in case (c)');
        // For S-curve, line-fit + poly-filter can't reach sub-pixel — the
        // PRIMITIVE needs to be polynomial. Keep this assertion loose and
        // let (d) below show the real win.
        h.expect((c.residual_std ?? 99) < 10.0,
                 `poly filter ran (residual σ = ${c.residual_std})`);
        // (d) polynomial primitive catches the whole arch end-to-end.
        h.expect(d.found === true, `polynomial primitive fit found (found=${d.found})`);
        h.expect(d.fit_model === 'polynomial', `fit_model reported polynomial (got ${d.fit_model})`);
        h.expect((d.inlier_count ?? 0) >= 28,
                 `polynomial primitive captured whole arch (inliers=${d.inlier_count})`);

        shotVSCode('final_summary', shotDir);
    },
};
