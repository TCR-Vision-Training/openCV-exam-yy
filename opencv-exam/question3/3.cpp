#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;


// ================================
// 全局变量：用于手工绘制椭圆真值
// ================================
Mat gtImage;
Point ellipseStart;
Point ellipseEnd;
bool drawingEllipse = false;
bool ellipseFinished = false;


// 鼠标绘制椭圆
void drawEllipse(int event, int x, int y, int flags, void*)
{
    if (event == EVENT_LBUTTONDOWN)
    {
        drawingEllipse = true;
        ellipseStart = Point(x, y);
        ellipseEnd = Point(x, y);
    }
    else if (event == EVENT_MOUSEMOVE && drawingEllipse)
    {
        ellipseEnd = Point(x, y);

        Mat temp = gtImage.clone();

        Rect box(
            min(ellipseStart.x, ellipseEnd.x),
            min(ellipseStart.y, ellipseEnd.y),
            abs(ellipseEnd.x - ellipseStart.x),
            abs(ellipseEnd.y - ellipseStart.y)
        );

        if (box.width > 0 && box.height > 0)
        {
            Point center(
                box.x + box.width / 2,
                box.y + box.height / 2
            );

            Size axes(
                box.width / 2,
                box.height / 2
            );

            ellipse(
                temp,
                center,
                axes,
                0,
                0,
                360,
                Scalar(0, 0, 255),
                2
            );
        }

        imshow("Draw Ground Truth", temp);
    }
    else if (event == EVENT_LBUTTONUP)
    {
        drawingEllipse = false;
        ellipseFinished = true;
        ellipseEnd = Point(x, y);
    }
}


// ================================
// 根据两个点生成椭圆真值
// ================================
Mat createGroundTruth(const Mat& image)
{
    Mat gt = Mat::zeros(image.size(), CV_8UC1);

    Rect box(
        min(ellipseStart.x, ellipseEnd.x),
        min(ellipseStart.y, ellipseEnd.y),
        abs(ellipseEnd.x - ellipseStart.x),
        abs(ellipseEnd.y - ellipseStart.y)
    );

    if (box.width <= 0 || box.height <= 0)
    {
        cerr << "椭圆区域无效！" << endl;
        return gt;
    }

    Point center(
        box.x + box.width / 2,
        box.y + box.height / 2
    );

    Size axes(
        box.width / 2,
        box.height / 2
    );

    ellipse(
        gt,
        center,
        axes,
        0,
        0,
        360,
        Scalar(255),
        FILLED
    );

    return gt;
}


// ================================
// GrabCut mask 转成二值前景
// ================================
Mat getForegroundMask(const Mat& grabMask)
{
    Mat foreground = Mat::zeros(grabMask.size(), CV_8UC1);

    foreground.setTo(
        255,
        (grabMask == GC_FGD) |
        (grabMask == GC_PR_FGD)
    );

    return foreground;
}


// ================================
// 计算 IoU
// ================================
double calculateIoU(const Mat& prediction, const Mat& groundTruth)
{
    CV_Assert(prediction.size() == groundTruth.size());

    Mat predBinary;
    Mat gtBinary;

    threshold(
        prediction,
        predBinary,
        0,
        255,
        THRESH_BINARY
    );

    threshold(
        groundTruth,
        gtBinary,
        0,
        255,
        THRESH_BINARY
    );

    Mat intersection;
    Mat unionMask;

    bitwise_and(
        predBinary,
        gtBinary,
        intersection
    );

    bitwise_or(
        predBinary,
        gtBinary,
        unionMask
    );

    double intersectionArea = countNonZero(intersection);
    double unionArea = countNonZero(unionMask);

    if (unionArea == 0)
        return 0.0;

    return intersectionArea / unionArea;
}


// ================================
// 形态学后处理
// ================================
Mat morphologyProcess(const Mat& input)
{
    Mat result = input.clone();

    // 椭圆结构元素
    Mat kernel = getStructuringElement(
        MORPH_ELLIPSE,
        Size(5, 5)
    );

    // 开运算：去掉小噪声
    morphologyEx(
        result,
        result,
        MORPH_OPEN,
        kernel
    );

    // 闭运算：填补小缺口
    morphologyEx(
        result,
        result,
        MORPH_CLOSE,
        kernel
    );

    return result;
}


// ================================
// 保留最大连通区域
// ================================
Mat keepLargestComponent(const Mat& input)
{
    Mat binary;
    threshold(
        input,
        binary,
        0,
        255,
        THRESH_BINARY
    );

    Mat labels;
    Mat stats;
    Mat centroids;

    int numLabels = connectedComponentsWithStats(
        binary,
        labels,
        stats,
        centroids,
        8,
        CV_32S
    );

    if (numLabels <= 1)
        return binary;

    int largestLabel = 1;
    int largestArea = stats.at<int>(1, CC_STAT_AREA);

    for (int i = 2; i < numLabels; i++)
    {
        int area = stats.at<int>(i, CC_STAT_AREA);

        if (area > largestArea)
        {
            largestArea = area;
            largestLabel = i;
        }
    }

    Mat result = Mat::zeros(binary.size(), CV_8UC1);

    result.setTo(
        255,
        labels == largestLabel
    );

    return result;
}


// ================================
// 主函数
// ================================
int main()
{
    // ==========================================
    // 1. 读取图像
    // ==========================================

    string filename = "../question3/picture3/image3.png";

    Mat image = imread(filename);

    if (image.empty())
    {
        cerr << "错误：无法读取图片！" << endl;
        cerr << "请检查 q3_grabcut.png 是否放在项目目录。" << endl;
        return -1;
    }

    cout << "图片读取成功！" << endl;
    cout << "图片大小："
         << image.cols
         << " × "
         << image.rows
         << endl;


    // ==========================================
    // 2. 矩形框初始化
    // ==========================================

    /*
        如果你想自己用鼠标框选：
        uncomment 下一行

        Rect rect = selectROI(
            "Select Object Rectangle",
            image
        );

        如果不想每次手动画，
        使用下面的自动矩形。
    */

    int x = static_cast<int>(image.cols * 0.28);
    int y = static_cast<int>(image.rows * 0.16);

    int width = static_cast<int>(image.cols * 0.47);
    int height = static_cast<int>(image.rows * 0.67);

    Rect rect(x, y, width, height);

    cout << endl;
    cout << "========== GrabCut 矩形初始化 ==========" << endl;

    cout << "矩形："
         << "x = " << rect.x
         << ", y = " << rect.y
         << ", width = " << rect.width
         << ", height = " << rect.height
         << endl;


    // 显示矩形
    Mat rectImage = image.clone();

    rectangle(
        rectImage,
        rect,
        Scalar(0, 0, 255),
        2
    );

    imshow(
        "Initial Rectangle",
        rectImage
    );

    waitKey(1000);


    // ==========================================
    // 3. GrabCut
    // ==========================================

    Mat mask(
        image.size(),
        CV_8UC1,
        Scalar(GC_BGD)
    );

    Mat bgdModel(
        1,
        65,
        CV_64FC1,
        Scalar(0)
    );

    Mat fgdModel(
        1,
        65,
        CV_64FC1,
        Scalar(0)
    );


    cout << endl;
    cout << "开始 GrabCut..." << endl;


    grabCut(
        image,
        mask,
        rect,
        bgdModel,
        fgdModel,
        8,
        GC_INIT_WITH_RECT
    );


    // ==========================================
    // 4. 得到前景掩膜
    // ==========================================

    Mat foregroundMask =
        getForegroundMask(mask);


    imwrite(
        "grabcut_mask.png",
        foregroundMask
    );


    // ==========================================
    // 5. 生成前景结果
    // ==========================================

    Mat foreground;

    image.copyTo(
        foreground,
        foregroundMask
    );

    imwrite(
        "grabcut_result.png",
        foreground
    );


    imshow(
        "GrabCut Mask",
        foregroundMask
    );

    imshow(
        "GrabCut Result",
        foreground
    );

    waitKey(1000);


    // ==========================================
    // 6. 手工绘制椭圆真值
    // ==========================================

    cout << endl;
    cout << "========== 绘制真值 GT ==========" << endl;
    cout << "请在图片上拖动鼠标画出目标椭圆。" << endl;
    cout << "画完后按 Enter。" << endl;

    gtImage = image.clone();

    namedWindow(
        "Draw Ground Truth"
    );

    setMouseCallback(
        "Draw Ground Truth",
        drawEllipse
    );

    imshow(
        "Draw Ground Truth",
        gtImage
    );

    while (true)
    {
        int key = waitKey(20);

        if (key == 13 && ellipseFinished)
            break;

        if (key == 27)
        {
            cout << "用户取消。" << endl;
            return 0;
        }
    }

    destroyWindow(
        "Draw Ground Truth"
    );


    // 创建 GT
    Mat groundTruth =
        createGroundTruth(image);


    imwrite(
        "ground_truth.png",
        groundTruth
    );


    // ==========================================
    // 7. 第一次 IoU
    // ==========================================

    double iou1 =
        calculateIoU(
            foregroundMask,
            groundTruth
        );

    cout << endl;
    cout << "========== 第一次分割结果 ==========" << endl;

    cout << "IoU = "
         << iou1
         << endl;


    // ==========================================
    // 8. 如果 IoU < 0.85
    //    进行二次 GrabCut
    // ==========================================

    Mat finalMask = foregroundMask.clone();

    if (iou1 < 0.85)
    {
        cout << endl;
        cout << "IoU < 0.85，开始进行二次 GrabCut..." << endl;


        // --------------------------------------
        // 根据第一次结果建立新的 GrabCut mask
        // --------------------------------------

        Mat refineMask(
            image.size(),
            CV_8UC1,
            Scalar(GC_BGD)
        );


        // 矩形外：确定背景
        refineMask.setTo(
            GC_BGD
        );


        // 矩形内部先设为可能前景
        refineMask(rect).setTo(
            GC_PR_FGD
        );


        // 第一次分割得到的前景
        // 设置成 probable foreground
        Mat insideForeground =
            foregroundMask(rect);

        refineMask(rect).setTo(
            GC_PR_FGD,
            insideForeground > 0
        );


        // 矩形外部保持背景


        // --------------------------------------
        // 第二次 GrabCut
        // --------------------------------------

        grabCut(
            image,
            refineMask,
            rect,
            bgdModel,
            fgdModel,
            8,
            GC_INIT_WITH_MASK
        );


        finalMask =
            getForegroundMask(
                refineMask
            );


        // --------------------------------------
        // 形态学处理
        // --------------------------------------

        finalMask =
            morphologyProcess(
                finalMask
            );


        // --------------------------------------
        // 最大连通区域
        // --------------------------------------

        finalMask =
            keepLargestComponent(
                finalMask
            );


        // --------------------------------------
        // 保存
        // --------------------------------------

        imwrite(
            "grabcut_refined_mask.png",
            finalMask
        );


        Mat refinedResult;

        image.copyTo(
            refinedResult,
            finalMask
        );

        imwrite(
            "grabcut_refined_result.png",
            refinedResult
        );


        // ======================================
        // 第二次 IoU
        // ======================================

        double iou2 =
            calculateIoU(
                finalMask,
                groundTruth
            );

        cout << endl;
        cout << "========== 改进后结果 ==========" << endl;

        cout << "IoU = "
             << iou2
             << endl;


        imshow(
            "Refined Mask",
            finalMask
        );

        imshow(
            "Refined Result",
            refinedResult
        );
    }
    else
    {
        cout << endl;
        cout << "IoU 已达到要求，无需进一步处理。" << endl;
    }


    // ==========================================
    // 9. 显示 GT
    // ==========================================

    imshow(
        "Ground Truth",
        groundTruth
    );


    cout << endl;
    cout << "========================================" << endl;
    cout << "程序运行完成！" << endl;
    cout << "输出文件：" << endl;
    cout << "  grabcut_mask.png" << endl;
    cout << "  grabcut_result.png" << endl;
    cout << "  ground_truth.png" << endl;
    cout << "  grabcut_refined_mask.png" << endl;
    cout << "  grabcut_refined_result.png" << endl;
    cout << "========================================" << endl;


    waitKey(0);

    destroyAllWindows();

    return 0;
}

