package com.pocketkitty.pocket_kitty

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Handler
import android.os.Looper
import androidx.exifinterface.media.ExifInterface
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors

/**
 * 口袋毛孩 · Android 原生桥接（完全离线版）
 *
 * 抠图引擎：TensorFlow Lite + U2-Net 轻量版（u2netp，320×320 输入）
 * - 模型打包进 APK 的 android 原生 assets（assets/ 根目录），运行时零网络请求
 * - 无 GMS 的华为手机可直接运行
 *
 * 【v1.0.1 修复】模型加载路径：
 *   上一版错误地按 "flutter_assets/u2netp.tflite" 查找——该前缀只对 pubspec.yaml
 *   声明的 Flutter 资源成立。模型实际在 android/app/src/main/assets/ 下，
 *   打包后位于 APK assets/ 根目录，应直接用文件名访问。
 *   现按多个候选路径依次尝试，全部失败时给出可操作提示。
 */
class MainActivity : FlutterActivity() {

    companion object {
        private const val CHANNEL = "pet_segmentation/segment"

        /** 构建标识：用于真机上确认装的是修复后的包（首页底部可见，报错时自动带上） */
        private const val BUILD_TAG = "v1.0.2-assetmgr"

        /** 模型文件名。想换全量版就下载 u2net.tflite 放进 assets 并改这里 */
        private const val MODEL_FILE = "u2netp.tflite"
        private const val MODEL_INPUT = 320

        /**
         * 模型候选路径，按顺序尝试：
         * 1. android 原生 assets 根目录（本工程的实际位置）
         * 2. pubspec 声明在仓库根目录的写法（flutter_assets/u2netp.tflite）
         * 3. pubspec 声明在 assets/models/ 的写法（flutter_assets/assets/models/u2netp.tflite）
         */
        private val MODEL_CANDIDATES = listOf(
            MODEL_FILE,
            "flutter_assets/$MODEL_FILE",
            "flutter_assets/assets/models/$MODEL_FILE"
        )

        /** U2-Net 标准预处理：ImageNet mean/std 归一化 */
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    private val mainHandler = Handler(Looper.getMainLooper())
    private val executor = Executors.newSingleThreadExecutor()
    private var interpreter: Interpreter? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "buildVersion" -> result.success(BUILD_TAG)
                    "removeBackground" -> {
                        val path = call.argument<String>("path")
                        if (path.isNullOrBlank()) {
                            result.error("BAD_ARGS", "缺少图片路径", null)
                        } else {
                            runSegmentation(path, result)
                        }
                    }
                    "clickSound" -> {
                        // 还没放 meow.mp3 时的临时叫声：系统提示音代替，不缺资源不崩
                        try {
                            val tg = ToneGenerator(AudioManager.STREAM_MUSIC, 80)
                            tg.startTone(ToneGenerator.TONE_PROP_BEEP, 150)
                            mainHandler.postDelayed({ tg.release() }, 400)
                            result.success(null)
                        } catch (_: Exception) {
                            result.success(null)
                        }
                    }
                    else -> result.notImplemented()
                }
            }
    }

    private fun runSegmentation(path: String, result: MethodChannel.Result) {
        executor.execute {
            try {
                val segmenter = obtainInterpreter()
                val bitmap = decodeScaled(path, maxDim = 2048)

                // 兼容 NCHW / NHWC 两种转换布局
                val inShape = segmenter.getInputTensor(0).shape()
                val input = buildInput(bitmap, nchw = inShape.size == 4 && inShape[1] == 3)

                // U2-Net 转换产物有 7 个输出（d1..d7 侧显著图），
                // 必须按数量分配输出缓冲，output[0] 即最终融合图
                val outputs = HashMap<Int, Any>()
                val buffers = Array(segmenter.outputTensorCount) { idx ->
                    val count = segmenter.getOutputTensor(idx).shape()
                        .fold(1) { acc, d -> acc * maxOf(d, 1) }
                    ByteBuffer.allocateDirect(count * 4)
                        .order(ByteOrder.nativeOrder())
                        .also { outputs[idx] = it }
                }

                segmenter.runForMultipleInputsOutputs(arrayOf(input), outputs)

                val first = buffers[0]
                first.rewind()
                val mask = FloatArray(MODEL_INPUT * MODEL_INPUT)
                for (i in mask.indices) mask[i] = first.float

                val outFile = composeAndSave(bitmap, mask)
                mainHandler.post { result.success(outFile.absolutePath) }
            } catch (e: Exception) {
                mainHandler.post {
                    // 报错自动带构建标识：真机上看到 v1.0.2-assetmgr 才说明装的是修复后的包
                    result.error(
                        "SEGMENT_FAIL",
                        "[${BUILD_TAG}] ${e.message ?: "抠图失败"}",
                        null
                    )
                }
            }
        }
    }

    /**
     * 懒加载模型：只初始化一次。
     * 用 Android 原生 AssetManager 读取（不是 Flutter 的 rootBundle——那是 Dart 侧的 API），
     * 依次尝试所有候选路径，全部失败时抛出带修复指引的明确错误。
     */
    private fun obtainInterpreter(): Interpreter {
        synchronized(this) {
            interpreter?.let { return it }

            val model = loadModelBuffer()
            val segmenter = Interpreter(model, Interpreter.Options().setNumThreads(4))

            // 动态 shape 的转换产物先固定到 320×320，再统一分配张量，
            // 保证后续查询输出 shape 可用
            val inShape = segmenter.getInputTensor(0).shape()
            if (inShape.any { it <= 0 }) {
                segmenter.resizeInput(0, intArrayOf(1, MODEL_INPUT, MODEL_INPUT, 3))
            }
            segmenter.allocateTensors()

            interpreter = segmenter
            return segmenter
        }
    }

    /** 从 APK assets 读取模型字节，候选路径逐一尝试 */
    private fun loadModelBuffer(): ByteBuffer {
        val tried = StringBuilder()
        for (candidate in MODEL_CANDIDATES) {
            try {
                val bytes = assets.open(candidate).use { it.readBytes() }
                val buffer = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
                buffer.put(bytes)
                buffer.rewind()
                return buffer
            } catch (_: Exception) {
                tried.append(candidate).append("  ")
            }
        }
        throw IllegalStateException(
            "抠图模型没有打进 APK（尝试过：$tried）。" +
                "修复方法：确认 android/app/src/main/assets/$MODEL_FILE 文件存在于仓库并重新推送打包；" +
                "云端打包流程检测到模型缺失时会自动从源头下载，无需手工处理。"
        )
    }

    /** 原图 → 320×320 → 归一化 FloatBuffer（自动适配 NCHW / NHWC） */
    private fun buildInput(bitmap: Bitmap, nchw: Boolean): ByteBuffer {
        val scaled = Bitmap.createScaledBitmap(bitmap, MODEL_INPUT, MODEL_INPUT, true)
        val pixels = IntArray(MODEL_INPUT * MODEL_INPUT)
        scaled.getPixels(pixels, 0, MODEL_INPUT, 0, 0, MODEL_INPUT, MODEL_INPUT)
        scaled.recycle()

        val buffer = ByteBuffer
            .allocateDirect(MODEL_INPUT * MODEL_INPUT * 3 * 4)
            .order(ByteOrder.nativeOrder())

        fun norm(channel: Int, ch: Int): Float =
            (channel / 255f - MEAN[ch]) / STD[ch]

        if (nchw) {
            for (ch in 0..2) {
                for (p in pixels) {
                    val c = when (ch) {
                        0 -> Color.red(p)
                        1 -> Color.green(p)
                        else -> Color.blue(p)
                    }
                    buffer.putFloat(norm(c, ch))
                }
            }
        } else {
            for (p in pixels) {
                buffer.putFloat(norm(Color.red(p), 0))
                buffer.putFloat(norm(Color.green(p), 1))
                buffer.putFloat(norm(Color.blue(p), 2))
            }
        }
        buffer.rewind()
        return buffer
    }

    /**
     * 输出 320×320 显著性图 → min-max 归一化 → 双线性放大回原尺寸 → 逐像素合成 alpha。
     * 不做硬阈值切割，毛发边缘按显著度半透明过渡，橘猫毛尖不会被切掉。
     */
    private fun composeAndSave(src: Bitmap, mask: FloatArray): File {
        val w = src.width
        val h = src.height

        var mn = Float.MAX_VALUE
        var mx = -Float.MAX_VALUE
        for (v in mask) {
            if (v < mn) mn = v
            if (v > mx) mx = v
        }
        val range = (mx - mn).coerceAtLeast(1e-6f)

        val maskBmp = Bitmap.createBitmap(MODEL_INPUT, MODEL_INPUT, Bitmap.Config.ARGB_8888)
        val maskPixels = IntArray(MODEL_INPUT * MODEL_INPUT) { i ->
            val a = (((mask[i] - mn) / range) * 255f).toInt().coerceIn(0, 255)
            Color.rgb(a, a, a)
        }
        maskBmp.setPixels(maskPixels, 0, MODEL_INPUT, 0, 0, MODEL_INPUT, MODEL_INPUT)
        val scaledMask = Bitmap.createScaledBitmap(maskBmp, w, h, true)

        val pixels = IntArray(w * h)
        src.getPixels(pixels, 0, w, 0, 0, w, h)
        val maskVals = IntArray(w * h)
        scaledMask.getPixels(maskVals, 0, w, 0, 0, w, h)

        val out = IntArray(w * h)
        var covered = 0f
        for (i in pixels.indices) {
            val a = Color.red(maskVals[i])
            if (a > 127) covered += 1f
            val p = pixels[i]
            out[i] = if (a == 0) 0
            else Color.argb(a, Color.red(p), Color.green(p), Color.blue(p))
        }

        // 前景占比过低 = 没认出主体，按失败处理，给出可读提示
        if (covered / (w * h) < 0.01f) {
            throw IllegalStateException("没有识别到明确的主体，试试更清晰、宠物占比更大的照片")
        }

        val resultBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        resultBmp.setPixels(out, 0, w, 0, 0, w, h)

        val dir = getExternalFilesDir(null) ?: filesDir
        val file = File(dir, "cutout_${System.currentTimeMillis()}.png")
        FileOutputStream(file).use { fos ->
            resultBmp.compress(Bitmap.CompressFormat.PNG, 100, fos)
        }

        resultBmp.recycle()
        scaledMask.recycle()
        maskBmp.recycle()
        return file
    }

    /** 解码 + 按最长边降采样 + 按 EXIF 方向转正 */
    private fun decodeScaled(path: String, maxDim: Int): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, bounds)
        var sample = 1
        var side = maxOf(bounds.outWidth, bounds.outHeight)
        while (side / 2 >= maxDim) {
            sample *= 2
            side /= 2
        }
        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        val bmp = BitmapFactory.decodeFile(path, opts)
            ?: throw IllegalStateException("不是有效的图片文件")

        val rotation = try {
            when (ExifInterface(path).getAttributeInt(
                ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL
            )) {
                ExifInterface.ORIENTATION_ROTATE_90 -> 90f
                ExifInterface.ORIENTATION_ROTATE_180 -> 180f
                ExifInterface.ORIENTATION_ROTATE_270 -> 270f
                else -> 0f
            }
        } catch (_: Exception) {
            0f
        }

        return if (rotation != 0f) {
            val matrix = Matrix().apply { postRotate(rotation) }
            Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
        } else {
            bmp
        }
    }

    override fun onDestroy() {
        executor.shutdown()
        interpreter?.close()
        interpreter = null
        super.onDestroy()
    }
}
