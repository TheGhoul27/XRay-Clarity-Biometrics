from pipeline.pipeline import single_pair
from pathlib import Path

single_pair(
    moving=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\data\\images\\135720-V0.bmp"),
    reference=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\code\\135644-V0.bmp"),
    out_dir=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\pipeline\\sample"),
    ckpt=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\yolo_type\\runs\\segment\\train\\weights\\best.pt"),
    imgsz=640,
    conf=0.1,
    device="cuda",
    class_filter={0}
)

single_pair(
    moving=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\data\\images\\135644-V0.bmp"),
    reference=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\code\\135644-V0.bmp"),
    out_dir=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\pipeline\\sample"),
    ckpt=Path("E:\\Masters_College_Work\\RA_CyLab\\X-Ray\\yolo_type\\runs\\segment\\train\\weights\\best.pt"),
    imgsz=640,
    conf=0.1,
    device="cuda",
    class_filter={0},
    dump_debug=True
)