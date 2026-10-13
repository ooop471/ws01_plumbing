

import argparse
import time
from pathlib import Path
#引入opencv中算法和numpy计算
import cv2
import numpy as np

#创建结构体
def parse_args():
    #提示作用：双目转深度
    p = argparse.ArgumentParser(description="SGBM stereo pair -> disparity/depth")
    #读取左右图路径
    p.add_argument("--left", required=True, help="左图路径")
    p.add_argument("--right", required=True, help="右图路径")
    #输出到out目录
    p.add_argument("--out", default="out", help="输出目录")
    p.add_argument("--num-disparities", type=int, default=192,
                   help="视差搜索范围，必须是 16 的倍数（默认 192）")
    #z=fb/d，设置数值
    p.add_argument("--block-size", type=int, default=5, help="SGBM 块大小，奇数（默认 5）")
    p.add_argument("--fx", type=float, default=1000.0, help="[示例] 焦距(像素)")
    p.add_argument("--baseline", type=float, default=0.1, help="[示例] 基线(米)")
    return p.parse_args()


def colorize_disp(disp, valid, max_disp):
    """把视差图染成彩色（无效区域涂黑），便于直观看图。"""
    d = np.clip(disp, 0, max_disp)
    vis = cv2.applyColorMap((255.0 * d / max_disp).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[~valid] = 0
    return vis


def colorize_depth(depth, valid, p_lo=2, p_hi=98):
    """深度图伪彩（近处暖色/远处冷色）。"""
    dv = depth[valid]
    if dv.size == 0:
        return np.zeros((*depth.shape, 3), np.uint8)
    lo, hi = np.percentile(dv, [p_lo, p_hi])
    hi = max(hi, lo + 1e-6)
    # 反转：近(小深度) -> 亮
    norm = np.clip((hi - depth) / (hi - lo), 0, 1)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    vis[~valid] = 0
    return vis


def main():
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    #调用opencv读取左右图
    imgL = cv2.imread(a.left, cv2.IMREAD_COLOR)
    imgR = cv2.imread(a.right, cv2.IMREAD_COLOR)
    #读图失败报错
    if imgL is None or imgR is None:
        raise SystemExit(f"读图失败: {a.left} / {a.right}")
    if imgL.shape != imgR.shape:
        raise SystemExit(f"左右图尺寸不一致: {imgL.shape} vs {imgR.shape}")
    print(f"输入尺寸: {imgL.shape[1]}x{imgL.shape[0]}")
    #==============================================================================
#采用SGBM算法计算视差即Z=fb/d，反过来求d   （psm，raft，cre，unimatch）
    #opencv读取的图是BGR格式，SGBM需要灰度图转换
    gL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    gR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    #视差搜索范围定义，16取正
    nd = max(16, (a.num_disparities // 16) * 16)
    #SGBM大小
    bs = a.block_size if a.block_size % 2 == 1 else a.block_size + 1
    # P1/P2 惩罚项按 OpenCV 推荐公式取
    p1, p2 = 8 * 1 * bs * bs, 32 * 1 * bs * bs

    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=nd,
        blockSize=bs,
        P1=p1,
        P2=p2,
        #左右一致性检查阈值。SGBM 可以根据左右匹配结果判断某些视差是否不可靠
        disp12MaxDiff=1,
        #唯一性约束。如果最佳匹配和次佳匹配太接近，说明匹配不可靠，该像素会被判为无效
        uniquenessRatio=10,
        #用于过滤小面积的噪声斑点
        speckleWindowSize=100,
        #控制斑点区域内允许的视差变化范围
        speckleRange=2,
        #没看懂，当规定值用
        preFilterCap=63,
        #优化版sgbm
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    #==============================================================================
    #跑了多久
    t0 = time.perf_counter()
    disp16 = sgbm.compute(gL, gR)              # 定点: 真实视差 = 值/16
    t_sgbm = time.perf_counter() - t0
    #WLS滤波处理
    has_wls = hasattr(cv2, "ximgproc")  # 模块检测
    disp_wls16 = None
    t_wls = 0.0
    if has_wls:
        try:
            right_matcher = cv2.ximgproc.createRightMatcher(sgbm)
            t1 = time.perf_counter()
            #叫gr，gl是因为sgbm是计算灰度的
            dispR16 = right_matcher.compute(gR, gL)# 定点: 真实视差 = 值/16
            wls = cv2.ximgproc.createDisparityWLSFilter(sgbm)
            wls.setLambda(8000)
            wls.setSigmaColor(1.5)
            #得到滤波后的视差图
            disp_wls16 = wls.filter(disp16, gL, None, dispR16)
            t_wls = time.perf_counter() - t1
        #跑崩退出
        except Exception as e:                                  # noqa: BLE001
            print(f"[警告] WLS 后处理失败，跳过: {e}")
            has_wls = False
    #转换为浮点数，方便后续计算
    def to_float(d16):
        d = d16.astype(np.float32)
        valid = d16 > (0 + 16 * 0)          # >0 即有效（OpenCV 用负值/0 表示无效）
        valid &= d16 != sgbm.getMinDisparity() * 16
        return d / 16.0, valid & (d16 > 0)
    #成功了用WLS的结果，否则用原始SGBM结果
    disp, valid = to_float(disp16)
    if has_wls and disp_wls16 is not None:
        disp_w, valid_w = to_float(disp_wls16)
    else:
        disp_w, valid_w = disp, valid

    # ---- 视差 -> 深度 ----依旧公式
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(valid_w & (disp_w > 0), a.fx * a.baseline / np.maximum(disp_w, 1e-6), np.nan)
    depth[~np.isfinite(depth)] = np.nan
    dvalid = valid_w & (disp_w > 0)

    # ---- 输出 ----
    #01视差02滤波视差03深度
    mx = float(nd)
    cv2.imwrite(str(out / "01_disparity_sgbm.png"), colorize_disp(disp, valid, mx))
    cv2.imwrite(str(out / "02_disparity_sgbm_wls.png"), colorize_disp(disp_w, valid_w, mx))
    cv2.imwrite(str(out / "03_depth_colormap.png"), colorize_depth(depth, dvalid))
    np.save(out / "disparity_sgbm_wls.npy", disp_w)
    np.save(out / "depth_meters.npy", depth)

    # ---- 统计 ----
    total = disp.size
    dv = disp_w[dvalid]
    print("-" * 58)
    print(f"SGBM 耗时        : {t_sgbm*1000:.0f} ms")
    if t_wls:
        print(f"WLS 后处理耗时    : {t_wls*1000:.0f} ms")
    print(f"有效像素占比      : {100.0*valid_w.sum()/total:.1f}% (WLS)")
    if dv.size:
        print(f"视差范围          : {dv.min():.1f} ~ {dv.max():.1f} px "
              f"(中位 {np.median(dv):.1f})")
    if dvalid.sum():
        dd = depth[dvalid]
        print(f"深度范围[示例内参]: {np.nanmin(dd):.2f} ~ {np.nanmax(dd):.2f} m "
              f"(中位 {np.median(dd):.2f})")
    print(f"输出目录          : {out.resolve()}")
    print("-" * 58)
    print("注意: 深度数值依赖 --fx/--baseline（当前为示例占位值），"
          "视差才是无需标定的直接结果。")


if __name__ == "__main__":
    main()
