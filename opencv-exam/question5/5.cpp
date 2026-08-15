// 第5题 频域滤波 —— 周期性条纹(扫描线纹)干扰消除（交互式）
//
// 流程：
//   1. 灰度图乘以 (-1)^(x+y) 实现频谱中心化，再做二维 DFT
//   2. 计算对数幅值谱并显示/保存；条纹对应一对共轭亮峰
//   3. 定位亮峰（自动检测 / 参考图差值谱检测 / 命令行指定 / 鼠标点击）
//   4. 构造高斯陷波滤波器，每个峰与其共轭峰成对抑制
//   5. 逆 DFT 还原得到去条纹图像，自动保存全部结果图
//
// 峰值定位优先顺序：
//   a) 命令行手动指定（最可靠）
//   b) 参考图差值谱检测：|含噪图 - 参考图| 的频谱里条纹最干净，
//      检测到的峰基本就是真正的条纹峰（推荐，需 --ref 指定参考图）
//   c) 含噪图像本身的频谱自动检测（无参考图时）
//   之后仍可在窗口里用鼠标左键点击亮峰修正
//
// 运行示例：
//   ./question5 image51.png --ref image52.png --save
//   ./question5 image51.png u1 v1 u2 v2 ... --save
//
// 交互操作（在“对数幅值谱”窗口，仅非 --save 模式）：
//   左键点击亮峰    添加/移除该陷波对（再点一次即撤销）
//   右键            清空所有陷波
//   ] 或 +          加大 sigma（条纹变浅但没干净时用）
//   [ 或 -          减小 sigma（图像被磨糊时用）
//   ESC             退出
// ============================================================

#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// ---------- 可调参数 ----------
const int AUTO_PAIRS    = 5;    // 自动检测的亮峰对数（基频+谐波）
const int DC_RADIUS     = 10;   // 排除直流分量附近的搜索半径（条纹峰常离中心很近）
const int PROTECT_R     = 12;   // 保护半径：陷波绝不允许吃掉中心直流（防画面变黑）
const float MIN_PEAK_VAL = 10.0f;  // 峰值幅度下限
const float CONJ_RATIO  = 0.3f; // 共轭点亮度须达峰值的该比例（近直流峰因叠加背景
                                // 不对称，放宽到 0.3）

// ---------- 全局状态（供鼠标回调使用） ----------
static Mat g_planes[2];     // 中心化频谱的实部/虚部
static Mat g_sign;          // (-1)^(x+y) 中心化因子
static Mat g_mag;           // 幅值谱
static Mat g_visBase;       // 灰度版对数幅值谱（用于画标记）
static Mat g_ref;           // 参考图
static Mat g_src;           // 原图（灰度，用于拼对比图）
static vector<Point> g_peaks;
static int g_cx = 0, g_cy = 0;
static int g_sigma = 5;
static bool g_hasRef = false;
static bool g_headless = false;   // 无图形界面时只保存图片，不弹窗
static double g_psnrBefore = 0.0; // 去条纹前 PSNR（原图 vs 参考图）

// 共轭峰：实图像频谱关于原点共轭对称，中心化后即 (W-x, H-y)（取模，
// 兼容奇数尺寸；奇数尺寸时不能简单用 2*中心-x）
static Point conjugateOf(Point p, int w, int h) {
    return Point((w - p.x) % w, (h - p.y) % h);
}

// 在陷波掩膜上对 (p) 位置加一个高斯陷波（1 = 保留，0 = 完全抑制）
static void addNotch(Mat& mask, Point p, int sigma) {
    int r = 3 * sigma;
    for (int i = max(0, p.y - r); i <= min(mask.rows - 1, p.y + r); ++i) {
        for (int j = max(0, p.x - r); j <= min(mask.cols - 1, p.x + r); ++j) {
            float dx = float(j - p.x), dy = float(i - p.y);
            float g = 1.0f - expf(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
            mask.at<float>(i, j) *= g;
        }
    }
}

static double psnr(const Mat& a, const Mat& b) {
    Mat d;
    absdiff(a, b, d);
    d.convertTo(d, CV_32F);
    double mse = mean(d.mul(d))[0];
    if (mse < 1e-8) return 999.0;
    return 10.0 * log10(255.0 * 255.0 / mse);
}

// 用当前陷波列表重新滤波并刷新所有窗口
static void rebuild() {
    Mat notchMask = Mat::ones(g_mag.size(), CV_32F);
    int px = g_mag.cols / 2, py = g_mag.rows / 2;
    for (Point p : g_peaks) {
        if (p.x < 0 || p.x >= g_mag.cols || p.y < 0 || p.y >= g_mag.rows) continue;
        if (norm(p - Point(px, py)) <= PROTECT_R) continue;   // 靠近中心的一律跳过
        addNotch(notchMask, p, g_sigma);
    }
    // 保护直流区域：即使误点/误检到中心附近，也强制恢复为保留
    circle(notchMask, Point(px, py), PROTECT_R, Scalar(1.0f), FILLED);

    Mat planesF[2];
    multiply(g_planes[0], notchMask, planesF[0]);
    multiply(g_planes[1], notchMask, planesF[1]);
    Mat filtered;
    merge(planesF, 2, filtered);

    Mat result;
    idft(filtered, result, DFT_INVERSE | DFT_SCALE | DFT_REAL_OUTPUT);
    multiply(result, g_sign, result);            // 还原中心化
    Mat out8;
    result.convertTo(out8, CV_8U);               // 自动截断到 [0,255]
    imwrite("q5_restored.png", out8);

    Mat notchVis;
    notchMask.convertTo(notchVis, CV_8U, 255.0);
    imwrite("q5_notch_mask.png", notchVis);

    // PSNR 前后对比（需要参考图）
    double psnrAfter = g_hasRef ? psnr(out8, g_ref) : 0.0;
    if (g_hasRef) {
        ofstream f("q5_psnr.txt");
        f << "去条纹前 PSNR: " << g_psnrBefore << " dB" << endl;
        f << "去条纹后 PSNR: " << psnrAfter << " dB" << endl;
        f << "PSNR 提升量: " << (psnrAfter - g_psnrBefore) << " dB" << endl;
        f << "陷波对数: " << g_peaks.size() / 2
          << ", sigma: " << g_sigma << endl;
        f.close();
    }

    if (!g_headless) {
        imshow("陷波掩膜", notchVis);
        imshow("去条纹结果", out8);
    }

    // ---------- 保存四格对比图（提交报告用） ----------
    Mat srcBgr, outBgr, maskBgr, specBgr;
    cvtColor(g_src, srcBgr, COLOR_GRAY2BGR);
    cvtColor(out8, outBgr, COLOR_GRAY2BGR);
    cvtColor(notchVis, maskBgr, COLOR_GRAY2BGR);
    cvtColor(g_visBase, specBgr, COLOR_GRAY2BGR);
    for (Point p : g_peaks)
        circle(specBgr, p, 10, Scalar(0, 0, 255), 2);
    putText(srcBgr, "Original", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.9,
            Scalar(0, 0, 255), 2);
    putText(outBgr, "Restored", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.9,
            Scalar(0, 255, 0), 2);
    putText(specBgr, "Log Spectrum", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.9,
            Scalar(0, 255, 0), 2);
    putText(maskBgr, "Notch Mask", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.9,
            Scalar(0, 255, 0), 2);
    Mat top, bottom, compare;
    hconcat(srcBgr, outBgr, top);
    hconcat(specBgr, maskBgr, bottom);
    vconcat(top, bottom, compare);
    imwrite("q5_compare.png", compare);

    cout << "当前 " << g_peaks.size() / 2 << " 对陷波, sigma=" << g_sigma;
    if (g_hasRef) {
        cout << ", PSNR提升=" << (psnrAfter - g_psnrBefore)
             << " dB (" << g_psnrBefore << " -> " << psnrAfter << ")"
             << ", 结果已写入 q5_psnr.txt";
    }
    cout << endl;
}

// 打印幅度最大的候选峰（含共轭与幅度），用于人工核对真正的条纹峰
static void printTopPeaks(const Mat& mag, int n = 12) {
    int cx = mag.cols / 2, cy = mag.rows / 2;
    Mat work = mag.clone();
    Mat mask = Mat::ones(mag.size(), CV_8U);
    circle(mask, Point(cx, cy), DC_RADIUS, Scalar(0), FILLED);
    cout << "原谱候选峰(幅度从大到小):" << endl;
    for (int i = 0; i < n; ++i) {
        double v;
        Point p;
        minMaxLoc(work, nullptr, &v, nullptr, &p, mask);
        if (v < MIN_PEAK_VAL) break;
        Point c = conjugateOf(p, mag.cols, mag.rows);
        cout << "  (" << p.x << "," << p.y << ") amp=" << (long)v
             << "   共轭(" << c.x << "," << c.y << ") amp="
             << (long)mag.at<float>(c) << endl;
        int rr = max(3, DC_RADIUS / 3);
        circle(work, p, rr, Scalar(0), FILLED);
        circle(work, c, rr, Scalar(0), FILLED);
    }
}

// 在谱图上重画陷波标记
static void showPeaksOnSpectrum() {
    Mat visColor;
    cvtColor(g_visBase, visColor, COLOR_GRAY2BGR);
    for (Point p : g_peaks)
        circle(visColor, p, 10, Scalar(0, 0, 255), 2);
    if (!g_headless)
        imshow("对数幅值谱(左键点峰/右键清空)", visColor);
    imwrite("q5_spectrum_peaks.png", visColor);
}

// 鼠标交互：左键添加/移除陷波对，右键清空
static void onMouse(int event, int x, int y, int, void*) {
    if (g_cx == 0) return;
    if (event == EVENT_LBUTTONDOWN) {
        Point p(x, y), c = conjugateOf(p, g_mag.cols, g_mag.rows);
        if (norm(p - Point(g_cx, g_cy)) <= PROTECT_R ||
            norm(c - Point(g_cx, g_cy)) <= PROTECT_R) {
            cout << "该点距直流中心太近（<" << PROTECT_R
                 << "px），已忽略（防止画面变黑），请点击离中心更远的亮峰" << endl;
            return;
        }
        auto hitP = [&](const Point& q) { return norm(q - p) <= 8.0; };
        auto hitC = [&](const Point& q) { return norm(q - c) <= 8.0; };
        bool hit = false;
        for (const Point& q : g_peaks)
            if (hitP(q) || hitC(q)) { hit = true; break; }

        if (hit) {
            vector<Point> keep;
            for (const Point& q : g_peaks)
                if (!hitP(q) && !hitC(q)) keep.push_back(q);
            g_peaks.swap(keep);
            cout << "已移除该陷波对" << endl;
        } else {
            g_peaks.push_back(p);
            if (norm(p - c) > 8.0) g_peaks.push_back(c);
            cout << "添加陷波: (" << p.x << "," << p.y
                 << ") + 共轭 (" << c.x << "," << c.y << ")" << endl;
        }
        showPeaksOnSpectrum();
        rebuild();
    } else if (event == EVENT_RBUTTONDOWN) {
        g_peaks.clear();
        cout << "已清空所有陷波" << endl;
        showPeaksOnSpectrum();
        rebuild();
    }
}

// 自动找峰值：排除直流区域，找幅度最大的点，并验证其共轭点同样明亮
static void detectPeaks(const Mat& mag, vector<Point>& peaks) {
    int cx = mag.cols / 2, cy = mag.rows / 2;
    Mat work = mag.clone();
    Mat mask = Mat::ones(mag.size(), CV_8U);
    circle(mask, Point(cx, cy), DC_RADIUS, Scalar(0), FILLED);

    for (int i = 0; i < AUTO_PAIRS; ++i) {
        double maxVal = 0;
        Point p;
        minMaxLoc(work, nullptr, &maxVal, nullptr, &p, mask);
        if (maxVal < MIN_PEAK_VAL) break;
        Point conj = conjugateOf(p, mag.cols, mag.rows);

        double conjVal = mag.at<float>(conj);
        if (conjVal >= CONJ_RATIO * maxVal) {
            peaks.push_back(p);
            peaks.push_back(conj);
        }
        // 无论是否采用，都屏蔽该邻域，继续找下一对
        circle(work, p, DC_RADIUS / 2, Scalar(0), FILLED);
        circle(work, conj, DC_RADIUS / 2, Scalar(0), FILLED);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "用法: " << argv[0] << " 图像路径 [峰坐标...] "
                "[--ref 参考图] [--sigma 1~40] [--save]" << endl;
        return 0;
    }

    // ---------- 解析命令行 ----------
    string refPath = "q5_reference.png";
    vector<string> numArgs;
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "--save") {
            g_headless = true;
        } else if (a == "--ref" && i + 1 < argc) {
            refPath = argv[++i];
        } else if (a == "--sigma" && i + 1 < argc) {
            g_sigma = max(1, min(40, atoi(argv[++i])));
        } else if (i >= 2) {
            numArgs.push_back(a);
        }
    }
    if (!g_headless && getenv("DISPLAY") == nullptr) {
        g_headless = true;
        cout << "未检测到图形显示环境，自动进入“只保存图片”模式。" << endl;
    }

    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    if (src.empty()) {
        cerr << "无法读取图像: " << argv[1] << endl;
        return -1;
    }
    g_src = src.clone();
    g_cx = src.cols / 2;
    g_cy = src.rows / 2;

    // ---------- 1. 频谱中心化 + 二维 DFT ----------
    Mat fsrc;
    src.convertTo(fsrc, CV_32F);
    g_sign = Mat::zeros(src.size(), CV_32F);   // (-1)^(x+y)
    for (int i = 0; i < src.rows; ++i)
        for (int j = 0; j < src.cols; ++j)
            g_sign.at<float>(i, j) = ((i + j) % 2 == 0) ? 1.0f : -1.0f;
    multiply(fsrc, g_sign, fsrc);

    Mat complexI;
    dft(fsrc, complexI, DFT_COMPLEX_OUTPUT);
    split(complexI, g_planes);

    // ---------- 2. 对数幅值谱 ----------
    magnitude(g_planes[0], g_planes[1], g_mag);
    Mat logMag;
    log(g_mag + 1.0f, logMag);
    double maxLog;
    minMaxLoc(logMag, nullptr, &maxLog);
    double cap = maxLog * 0.95;              // 压缩直流动态范围，让中高频峰可见
    logMag.setTo(cap, logMag > cap);
    normalize(logMag, g_visBase, 0, 255, NORM_MINMAX);
    g_visBase.convertTo(g_visBase, CV_8U);
    imwrite("q5_log_spectrum.png", g_visBase);
    printTopPeaks(g_mag, 12);

    // ---------- 3. 参考图（用于差值谱定位条纹峰 + PSNR） ----------
    g_ref = imread(refPath, IMREAD_GRAYSCALE);
    if (g_ref.empty()) {
        // 未指定或默认名不存在时，尝试输入图同目录下的常见名字
        std::filesystem::path inPath(argv[1]);
        vector<string> names = {"q5_reference.png", "reference.png",
                                "ref.png", "image52.png"};
        for (const string& nm : names) {
            std::filesystem::path cand = inPath.parent_path() / nm;
            g_ref = imread(cand.string(), IMREAD_GRAYSCALE);
            if (!g_ref.empty()) {
                cout << "自动找到参考图: " << cand.string() << endl;
                break;
            }
        }
    }
    g_hasRef = (!g_ref.empty() && g_ref.size() == src.size());
    if (g_ref.empty())
        cout << "未找到参考图（可用 --ref 指定路径），将用原图谱检测。" << endl;
    else if (!g_hasRef)
        cout << "参考图尺寸不一致，仅用于 PSNR 会跳过；差值谱检测不可用。" << endl;
    if (g_hasRef) {
        g_psnrBefore = psnr(src, g_ref);
        cout << "去条纹前 PSNR(原图 vs 参考图) = " << g_psnrBefore << " dB" << endl;
    }

    // ---------- 4. 定位条纹峰（优先级：手动 > 差值谱 > 原图谱） ----------
    if (!numArgs.empty()) {
        for (size_t i = 0; i + 1 < numArgs.size(); i += 2)
            g_peaks.push_back(Point(atoi(numArgs[i].c_str()), atoi(numArgs[i + 1].c_str())));
        size_t n = g_peaks.size();
        for (size_t i = 0; i < n; ++i)
            g_peaks.push_back(conjugateOf(g_peaks[i], src.cols, src.rows));
        cout << "手动指定峰值(共轭由程序补充):" << endl;
    } else if (g_hasRef) {
        // 差值谱：条纹在 |含噪图 - 参考图| 的频谱里最干净
        Mat diff;
        absdiff(src, g_ref, diff);
        Mat df;
        diff.convertTo(df, CV_32F);
        multiply(df, g_sign, df);
        Mat dc;
        dft(df, dc, DFT_COMPLEX_OUTPUT);
        Mat dp[2];
        split(dc, dp);
        Mat dmag;
        magnitude(dp[0], dp[1], dmag);
        detectPeaks(dmag, g_peaks);
        cout << "参考差值谱检测到的条纹峰(含共轭):" << endl;
    } else {
        detectPeaks(g_mag, g_peaks);
        cout << "原图谱自动检测到的峰值(含共轭):" << endl;
    }
    for (Point p : g_peaks)
        cout << "  (" << p.x << ", " << p.y << ")" << endl;

    // ---------- 5. 显示 + 处理 ----------
    showPeaksOnSpectrum();
    rebuild();

    if (g_headless) {
        cout << endl
             << "处理完成，所有输出图片已保存到:" << endl
             << "  " << std::filesystem::current_path().string() << endl
             << "  文件名: q5_log_spectrum.png / q5_spectrum_peaks.png" << endl
             << "          q5_notch_mask.png / q5_restored.png / q5_compare.png" << endl;
        return 0;
    }

    setMouseCallback("对数幅值谱(左键点峰/右键清空)", onMouse);

    cout << endl
         << "操作提示:" << endl
         << "  左键点击谱图亮峰 → 添加/移除陷波对，结果实时更新" << endl
         << "  右键 → 清空所有陷波" << endl
         << "  ] 或 + 加大 sigma，[ 或 - 减小 sigma，ESC 退出" << endl;

    while (true) {
        int key = waitKey(0) & 0xFF;
        if (key == 27) break;                     // ESC
        if (key == ']' || key == '+') {
            g_sigma = min(g_sigma + 1, 40);
            cout << "sigma = " << g_sigma << endl;
            rebuild();
        }
        if (key == '[' || key == '-') {
            g_sigma = max(g_sigma - 1, 1);
            cout << "sigma = " << g_sigma << endl;
            rebuild();
        }
    }

    destroyAllWindows();
    return 0;
}
