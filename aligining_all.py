# from pipeline.pipeline import batch_dir
# from pathlib import Path

# # AirPods 4 Backside
# batch_dir(
#     img_dir=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\data\\seg\\AirPods 4\\backside"),
#     ref_dir=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\data\\grade_a\\134503-V0.bmp"),
#     out_root=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\results\\AirPods 4\\backside"),
#     ckpt=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\yolo_type\\runs\\segment\\train\\weights\\best.pt"),
#     imgsz=640,
#     conf=0.1,
#     device="cuda",
#     class_filter={0},
#     dump_debug=True
# )

# # AirPods 4 Frontside
# batch_dir(
#     img_dir=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\data\\seg\\AirPods 4\\frontside"),
#     ref_dir=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\data\\grade_a\\134450-V0.bmp"),
#     out_root=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\results\\AirPods 4\\frontside"),
#     ckpt=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\yolo_type\\runs\\segment\\train\\weights\\best.pt"),
#     imgsz=640,
#     conf=0.1,
#     device="cuda",
#     class_filter={0},
#     dump_debug=True
# )

# # AirPods Max Backside
# batch_dir(
#     img_dir=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\data\\seg\\AirPods Max\\backside"),
#     ref_dir=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\data\\grade_a\\141154-V0.bmp"),
#     out_root=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\results\\AirPods Max\\backside"),
#     ckpt=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\yolo_type\\runs\\segment\\train\\weights\\best.pt"),
#     imgsz=640,
#     conf=0.1,
#     device="cuda",
#     class_filter={0},
#     dump_debug=True
# )

# # AirPods Max Frontside
# batch_dir(
#     img_dir=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\data\\seg\\AirPods Max\\frontside"),
#     ref_dir=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\data\\grade_a\\141140-V0.bmp"),
#     out_root=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\results\\AirPods Max\\frontside"),
#     ckpt=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\yolo_type\\runs\\segment\\train\\weights\\best.pt"),
#     imgsz=640,
#     conf=0.1,
#     device="cuda",
#     class_filter={0},
#     dump_debug=True
# )

# # AirPods Pro 2 Backside
# batch_dir(
#     img_dir=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\data\\seg\\AirPods Pro 2\\backside"),
#     ref_dir=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\data\\grade_a\\135734-V0.bmp"),
#     out_root=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\results\\AirPods Pro 2\\backside"),
#     ckpt=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\yolo_type\\runs\\segment\\train\\weights\\best.pt"),
#     imgsz=640,
#     conf=0.1,
#     device="cuda",
#     class_filter={0},
#     dump_debug=True
# )

# # AirPods Pro 2 Frontside
# batch_dir(
#     img_dir=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\data\\seg\\AirPods Pro 2\\frontside"),
#     ref_dir=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\data\\grade_a\\135644-V0.bmp"),
#     out_root=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\results\\AirPods Pro 2\\frontside"),
#     ckpt=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\yolo_type\\runs\\segment\\train\\weights\\best.pt"),
#     imgsz=640,
#     conf=0.1,
#     device="cuda",
#     class_filter={0},
#     dump_debug=True
# )

# # Galaxy Buds3 Pro Backside
# batch_dir(
#     img_dir=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\data\\seg\\Galaxy Buds3 Pro\\backside"),
#     ref_dir=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\data\\grade_a\\142152-V0.bmp"),
#     out_root=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\results\\Galaxy Buds3 Pro\\backside"),
#     ckpt=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\yolo_type\\runs\\segment\\train\\weights\\best.pt"),
#     imgsz=640,
#     conf=0.1,
#     device="cuda",
#     class_filter={0},
#     dump_debug=True
# )

# # Galaxy Buds3 Pro Frontside
# batch_dir(
#     img_dir=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\data\\seg\\Galaxy Buds3 Pro\\frontside"),
#     ref_dir=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\data\\grade_a\\142057-V0.bmp"),
#     out_root=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\results\\Galaxy Buds3 Pro\\frontside"),
#     ckpt=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\yolo_type\\runs\\segment\\train\\weights\\best.pt"),
#     imgsz=640,
#     conf=0.1,
#     device="cuda",
#     class_filter={0},
#     dump_debug=True
# )

# # Over Ear Backside
# batch_dir(
#     img_dir=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\data\\seg\\Over Ear\\backside"),
#     ref_dir=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\data\\grade_a\\143745-V0.bmp"),
#     out_root=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\results\\Over Ear\\backside"),
#     ckpt=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\yolo_type\\runs\\segment\\train\\weights\\best.pt"),
#     imgsz=640,
#     conf=0.1,
#     device="cuda",
#     class_filter={0},
#     dump_debug=True
# )

# # Over Ear Frontside
# batch_dir(
#     img_dir=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\data\\seg\\Over Ear\\frontside"),
#     ref_dir=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\data\\grade_a\\143633-V0.bmp"),
#     out_root=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\results\\Over Ear\\frontside"),
#     ckpt=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\yolo_type\\runs\\segment\\train\\weights\\best.pt"),
#     imgsz=640,
#     conf=0.1,
#     device="cuda",
#     class_filter={0},
#     dump_debug=True
# )

# dual_batch_runner.py
# -----------------------------------------------------------
# Drive pipeline.batch_dir_dual over all AirPods / Buds / Over-Ear
# datasets, re-using a single front & back template per product.
# -----------------------------------------------------------
import shutil
from pathlib import Path
from pipeline import batch_dir_dual


# ─────────────────────── helper ───────────────────────
def run_dual_job(img_dir: Path,
                 front_tpl: Path,
                 back_tpl:  Path,
                 out_root:  Path,
                 ckpt:      Path,
                 *,
                 imgsz: int        = 640,
                 conf:  float      = 0.1,
                 device: str       = "cuda",
                 class_filter: set = None,
                 dump_debug: bool  = False,
                 angle_step: int   = 5, 
                 score_mode: str   = "hybrid"):
    """
    1.  Clone `front_tpl` & `back_tpl` into two temp directories so each
        image in `img_dir` has a same-named reference file.
    2.  Call `batch_dir_dual` with those directories.
    """
    out_root.mkdir(parents=True, exist_ok=True)

    tmp_front = out_root / "_refs_front"
    tmp_back  = out_root / "_refs_back"
    tmp_front.mkdir(exist_ok=True)
    tmp_back.mkdir(exist_ok=True)

    # Copy (if missing) the single template so every file matches.
    for img_path in sorted(img_dir.glob("*.bmp")):        # adjust suffix if needed
        dst_f = tmp_front / img_path.name
        dst_b = tmp_back  / img_path.name
        if not dst_f.exists():
            shutil.copy2(front_tpl, dst_f)
        if not dst_b.exists():
            shutil.copy2(back_tpl, dst_b)

    batch_dir_dual(
        img_dir   = img_dir,
        front_dir = tmp_front,
        back_dir  = tmp_back,
        out_root  = out_root,
        ckpt      = ckpt,
        imgsz     = imgsz,
        conf      = conf,
        device    = device,
        class_filter = class_filter,
        dump_debug   = dump_debug,
        angle_step   = angle_step,
        score_mode   =  score_mode
    )


# ─────────────────────── common settings ───────────────────────
CKPT = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\yolo_type\runs\segment\train\weights\best.pt")
COMMON = dict(imgsz=640, conf=0.1, device="cuda",
              class_filter={0}, dump_debug=True, angle_step=90, score_mode="rmse")

# ─────────────────────── jobs list ───────────────────────
jobs = [

    # AirPods 4
    dict(name="AirPods 4 backside",
         img_dir = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\seg\AirPods 4\backside"),
         front_tpl = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\grade_a\134450-V0.bmp"),
         back_tpl  = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\grade_a\134503-V0.bmp"),
         out_root  = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\results\AirPods 4\backside")),

    dict(name="AirPods 4 frontside",
         img_dir = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\seg\AirPods 4\frontside"),
         front_tpl = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\grade_a\134450-V0.bmp"),
         back_tpl  = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\grade_a\134503-V0.bmp"),
         out_root  = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\results\AirPods 4\frontside")),

    # AirPods Max
    dict(name="AirPods Max backside",
         img_dir   = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\seg\AirPods Max\backside"),
         front_tpl = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\grade_a\141140-V0.bmp"),
         back_tpl  = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\grade_a\141154-V0.bmp"),
         out_root  = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\results\AirPods Max\backside")),

    dict(name="AirPods Max frontside",
         img_dir   = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\seg\AirPods Max\frontside"),
         front_tpl = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\grade_a\141140-V0.bmp"),
         back_tpl  = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\grade_a\141154-V0.bmp"),
         out_root  = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\results\AirPods Max\frontside")),

    # AirPods Pro 2
    dict(name="AirPods Pro 2 backside",
         img_dir   = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\seg\AirPods Pro 2\backside"),
         front_tpl = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\grade_a\135644-V0.bmp"),
         back_tpl  = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\grade_a\135734-V0.bmp"),
         out_root  = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\results\AirPods Pro 2\backside")),

    dict(name="AirPods Pro 2 frontside",
         img_dir   = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\seg\AirPods Pro 2\frontside"),
         front_tpl = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\grade_a\135644-V0.bmp"),
         back_tpl  = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\grade_a\135734-V0.bmp"),
         out_root  = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\results\AirPods Pro 2\frontside")),

    # Galaxy Buds3 Pro
    dict(name="Galaxy Buds3 Pro backside",
         img_dir   = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\seg\Galaxy Buds3 Pro\backside"),
         front_tpl = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\grade_a\142057-V0.bmp"),
         back_tpl  = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\grade_a\142152-V0.bmp"),
         out_root  = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\results\Galaxy Buds3 Pro\backside")),

    dict(name="Galaxy Buds3 Pro frontside",
         img_dir   = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\seg\Galaxy Buds3 Pro\frontside"),
         front_tpl = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\grade_a\142057-V0.bmp"),
         back_tpl  = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\grade_a\142152-V0.bmp"),
         out_root  = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\results\Galaxy Buds3 Pro\frontside")),

    # Over-Ear
    dict(name="Over Ear backside",
         img_dir   = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\seg\Over Ear\backside"),
         front_tpl = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\grade_a\143633-V0.bmp"),
         back_tpl  = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\grade_a\143745-V0.bmp"),
         out_root  = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\results\Over Ear\backside")),

    dict(name="Over Ear frontside",
         img_dir   = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\seg\Over Ear\frontside"),
         front_tpl = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\grade_a\143633-V0.bmp"),
         back_tpl  = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\data\grade_a\143745-V0.bmp"),
         out_root  = Path(r"E:\Masters_College_Work\RA_CyLab\X-Ray\results\Over Ear\frontside")),
]

# ─────────────────────── run all jobs ───────────────────────
for j in jobs:
    print(f"\n=== {j['name']} ===")
    run_dual_job(
        img_dir     = j['img_dir'],
        front_tpl   = j['front_tpl'],
        back_tpl    = j['back_tpl'],
        out_root    = j['out_root'],
        ckpt        = CKPT,
        **COMMON
    )
