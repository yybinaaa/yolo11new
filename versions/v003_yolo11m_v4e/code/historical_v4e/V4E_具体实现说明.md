# V4E 具体实现说明

## 1. 方法定位

V4E 的完整名称是 **Weak-Evidence Amplified dual-scale SAHI V4E**。它不重新训练模型，也不修改 YOLO11m 网络，而是在已有 V3E 提交结果上增加一个弱证据复检分支。

当前线上结果：

| 指标 | V4E |
|---|---:|
| Recall / Score | 0.8380 / 83.80 |
| Precision | 0.2949 |
| F1 | 0.4362 |
| mAP@0.5 | 0.4943 |
| TP / FP / FN | 833 / 1992 / 161 |
| 最终检测框 | 2825 |

V4E 的目标是：不直接提交低置信框，而是把标准 SAHI 中已经出现、但没有达到正常提交阈值的响应当作疑似漏检线索，再通过局部放大和上下文视野重新确认。

## 2. 三个输入

1. 检测权重：`epoch170.pt`；
2. 首轮 SAHI 候选缓存：最低保留到 `conf=0.05`；
3. V3E anchor：首轮候选和 Adaptive Reslice 候选均使用阈值 `0.18`，经过 class-wise NMS 后得到的 2570 个基础框。

V4E 始终完整保留 V3E 的 2570 个框，只允许在其后追加新框。

## 3. 完整处理流程

```text
标准 SAHI 候选缓存（conf >= 0.05）
    |
    |-- 仅选择 0.05 <= score < 0.18 的弱候选
    |
    |-- 排除已被 V3E 同类别框解释的候选
    |      条件：same image + same class + IoU >= 0.30
    |
    |-- 同类别空间聚类
    |      IoU >= 0.25 或 IoM >= 0.55
    |
    |-- 根据分数和多视野支持度排序
    |      priority = max_score + 0.025 * min(3, support - 1)
    |
    |-- 每张图最多选择 2 个疑似区域
    |      中心距离 <= 192 px 的簇合并到同一区域
    |
    |-- 每个区域生成两种原图裁剪
    |      640 x 640：zoom view
    |      1280 x 1280：context view
    |
    |-- 两种裁剪均运行同一个 epoch170.pt
    |      imgsz=1024, conf=0.05, batch=4, max_det=1000
    |
    |-- 把裁剪坐标映射回原图
    |
    |-- V4E 任一视野确认
    |      detection score >= 0.10
    |      detection class == weak seed class
    |      IoU(detection, seed) >= 0.10
    |          或 IoM(detection, seed) >= 0.50
    |
    |-- 对确认框执行 class-wise NMS，IoU=0.50
    |
    |-- anchor-preserving safe-add
           保留全部 V3E 框
           仅追加 same image + same class 下 IoU < 0.50 的新框
    |
    `-- 生成 V4E：2825 框 = 2570 anchor + 255 新框
```

## 4. 弱候选筛选

首先按 `(image_id, category_name)` 为 V3E 建立同类 anchor 索引。对首轮 SAHI 缓存中的每个候选执行：

```python
if not 0.05 <= candidate.score < 0.18:
    discard()

same_class_anchors = anchors[(image_id, class_name)]
if any(IoU(candidate.box, anchor.box) >= 0.30
       for anchor in same_class_anchors):
    discard()  # 已被 V3E 解释
else:
    keep_as_weak_evidence()
```

这里的弱候选只是复检种子，不会被直接写入提交文件。

## 5. 同类别空间聚类和区域选择

候选先按置信度降序排列。两个候选只有类别相同，并且满足以下任一条件时，才认为来自同一个弱证据区域：

```text
IoU >= 0.25
或
intersection / min(area_a, area_b) >= 0.55
```

其中第二项是 IoM，用于识别一个小框被另一个大框包含的情况。

每个簇的代表框是最先进入该簇的最高分框。支持度按不同 `(source, tile_origin)` 的数量计算：

```python
support = number_of_unique_views
priority = max_score + 0.025 * min(3, support - 1)
```

因此，同一位置在多个切片或全图分支中重复出现，会获得更高优先级。随后每张图最多选择 2 个区域；如果新簇中心距离已选区域不超过 192 像素，它会作为另一个 seed 合并到该区域。

## 6. 双视野裁剪

对每个区域中心 `(cx, cy)` 生成两个以该点为中心的裁剪：

```python
zoom_origin = clip((cx - 640 / 2,  cy - 640 / 2),  image_bounds)
context_origin = clip((cx - 1280 / 2, cy - 1280 / 2), image_bounds)

zoom_crop = image[y:y+640, x:x+640]
context_crop = image[y:y+1280, x:x+1280]
```

- `640` 视野提高小缺陷在模型输入中的相对尺寸；
- `1280` 视野保留更完整的周围结构，适合长条或依赖上下文的缺陷。

两个裁剪不是 ROIAlign，而是直接从原始图像切出，然后分别运行完整的 YOLO11m：

```python
results = model.predict(
    crops,
    imgsz=1024,
    conf=0.05,
    batch=4,
    device="0",
    max_det=1000,
    verbose=False,
)
```

模型输出框通过加回裁剪左上角坐标映射到原图：

```python
global_box = [
    local_x1 + origin_x,
    local_y1 + origin_y,
    local_x2 + origin_x,
    local_y2 + origin_y,
]
```

## 7. V4E 的确认规则

V4E 调用 `confirmed_records()` 时不指定 `view_type`，因此 `zoom` 和 `context` 两路候选都会参加确认。候选需要同时满足：

```python
detection.score >= 0.10
detection.class_id == seed.class_id
(
    IoU(detection.box, seed.box) >= 0.10
    or IoM(detection.box, seed.box) >= 0.50
)
```

之后把全部合格候选放在一起执行 `class-wise NMS(IoU=0.50)`。

需要特别区分：

- V4E：640 或 1280 **任一视野**确认即可；
- V4D：要求两个视野之间同类一致，并执行加权框融合；
- 因此 V4E 更激进，Recall 更高，FP 也更多。

## 8. Anchor-preserving safe-add

最终融合不重新对 V3E 和 V4E 候选进行全量联合 NMS，而是完整复制 V3E，再依次判断每个新增候选：

```python
output = list(v3e_anchor)
index = build_index(v3e_anchor, key=(image_id, category_name))

for candidate in v4e_confirmed_candidates:
    same_class_boxes = index[(candidate.image_id, candidate.category_name)]

    if any(IoU(candidate.box, box) >= 0.50 for box in same_class_boxes):
        continue

    output.append(candidate)
    same_class_boxes.append(candidate.box)
```

新增框一旦被接受，也会立即加入索引，所以后续重复新增框同样会被抑制。最终还会检查：

```python
output[:len(v3e_anchor)] == v3e_anchor
```

只要 V3E 前缀发生任何改变，程序就拒绝生成提交文件。

## 9. 当前运行统计

| 项目 | 数值 |
|---|---:|
| 测试图片 | 669 |
| 初始弱候选 | 4028 |
| 排除 anchor 已解释候选后 | 2320 |
| 空间簇 | 1629 |
| 实际规划图片 | 476 |
| 疑似区域 | 779 |
| 双视野任务 | 1558 |
| 双视野原始候选 | 2163 |
| V4 推理时间 | 96.92 s |
| V3E anchor | 2570 |
| V4E 新增 | 255 |
| V4E 总框数 | 2825 |

## 10. 参数表

| 参数 | V4E 数值 |
|---|---:|
| `weak_min` | 0.05 |
| `weak_max` | 0.18 |
| `zoom_size` | 640 |
| `context_size` | 1280 |
| `model_imgsz` | 1024 |
| `max_seeds_per_image` | 2 |
| `cluster_iou` | 0.25 |
| `cluster_iom` | 0.55 |
| `anchor_overlap_iou` | 0.30 |
| `dedupe_center_distance` | 192 px |
| `confirm_iou` | 0.10 |
| `confirm_iom` | 0.50 |
| `aggressive_threshold` | 0.10 |
| `nms_iou` | 0.50 |
| `novel_iou` | 0.50 |
| `candidate_conf` | 0.05 |
| `batch` | 4 |
| `max_det` | 1000 |

## 11. 一键复现

```powershell
.\.venv\Scripts\python.exe scripts\predict_weak_evidence_sahi_v4.py `
  --weights runs\yolo11\yolo11m_sahi_sf_continue_epoch150_to250\weights\epoch170.pt `
  --source "C:\Users\16125\Desktop\钢材AI\初赛测试集\初赛" `
  --first-pass-cache outputs\yolo11\epoch170_inference_ab_cache_conf005\epoch170_candidates_conf005.json `
  --anchor outputs\yolo11\epoch170_adaptive_reslice_v3_ab_5\05_V3E_epoch170_first018_ref018_union_nms050_submission.json `
  --output outputs\yolo11\epoch170_weak_evidence_sahi_v4 `
  --resume
```

默认参数已经对应当前 V4E 实验。`--resume` 会复用已经完成的双视野缓存，中断后再次执行同一条命令即可继续。

## 12. 源码对应关系

核心入口：`scripts/predict_weak_evidence_sahi_v4.py`

| 模块 | 函数 |
|---|---|
| 弱候选筛选与计划生成 | `build_weak_evidence_plan()` |
| 同类弱候选聚类 | `_cluster_weak_candidates()` |
| 中心裁剪起点计算 | `centered_origin()` |
| 双视野完整模型推理 | `run_dual_scale_inference()` |
| 裁剪框映射回原图 | `_append_result()` |
| 检测框与弱 seed 确认 | `detection_matches_seed()` |
| V4E 任一视野确认 | `confirmed_records()` |
| class-wise NMS | `src/inference/merger.py::classwise_nms()` |
| 安全追加 | `scripts/build_epoch170_adaptive_v3_ab.py::additive_safe_submission()` |
| V4E 变体定义 | `main()` 中的 `05_V4E_dual_view_any_t010_safe_add_submission` |

## 13. 最容易误解的三点

1. `0.05–0.18` 的弱候选不会直接提交，只用于确定在哪里重新观察；
2. `threshold=0.10` 是双视野复检结果的接受阈值，不是标准 SAHI 的最终阈值；
3. V4E 不要求 640 和 1280 同时确认，任一视野确认即可，因此它比 V4D 更偏向 Recall。

## 14. SAHI + V3E + V4E 完整伪代码

下面的伪代码对应当前最高分的完整链路。V4E 实际使用的 anchor 不是普通 SAHI 的直接输出，而是首轮 SAHI 与 Adaptive Reslice 联合形成的 V3E。

```text
ALGORITHM SAHI_V4E_INFERENCE(images, detector):

    CONSTANTS:
        TILE_SIZE              = 1024
        TILE_OVERLAP           = 0.20
        CANDIDATE_FLOOR        = 0.05
        NMS_IOU                = 0.50

        V3_FIRST_THRESHOLD     = 0.18
        V3_REFINE_THRESHOLD    = 0.18

        WEAK_MIN               = 0.05
        WEAK_MAX               = 0.18
        ANCHOR_EXPLAIN_IOU     = 0.30
        CLUSTER_IOU            = 0.25
        CLUSTER_IOM            = 0.55
        MAX_WEAK_REGIONS       = 2
        REGION_DEDUPE_DISTANCE = 192

        ZOOM_SIZE              = 640
        CONTEXT_SIZE           = 1280
        MODEL_IMGSZ            = 1024
        V4_ACCEPT_THRESHOLD    = 0.10
        CONFIRM_IOU            = 0.10
        CONFIRM_IOM            = 0.50
        NOVEL_IOU              = 0.50

    first_pass_cache = {}

    # ------------------------------------------------------------
    # Stage A: 标准 SAHI，生成首轮低阈值候选缓存
    # ------------------------------------------------------------
    FOR each image IN images:
        tile_candidates = []

        x_starts = GENERATE_STARTS(image.width,  TILE_SIZE, TILE_OVERLAP)
        y_starts = GENERATE_STARTS(image.height, TILE_SIZE, TILE_OVERLAP)

        FOR each y IN y_starts:
            FOR each x IN x_starts:
                tile = CROP(image, x, y, TILE_SIZE, TILE_SIZE)

                local_detections = detector.predict(
                    tile,
                    imgsz = 1024,
                    conf  = CANDIDATE_FLOOR
                )

                FOR each detection IN local_detections:
                    global_box = detection.box + [x, y, x, y]
                    tile_candidates.APPEND({
                        box: global_box,
                        score: detection.score,
                        class: detection.class,
                        source: "tile",
                        tile_origin: [x, y]
                    })

        IF number_of_tiles > 1:
            full_detections = detector.predict(
                image,
                imgsz = 1024,
                conf  = CANDIDATE_FLOOR
            )

            FOR each detection IN full_detections:
                tile_candidates.APPEND({
                    box: detection.box,
                    score: detection.score,
                    class: detection.class,
                    source: "full"
                })

        first_pass_cache[image.id] = tile_candidates

    # ------------------------------------------------------------
    # Stage B: Adaptive Reslice，生成重居中 1024 ROI 候选
    # ------------------------------------------------------------
    refinement_cache = {}

    FOR each image IN images:
        triggers = []

        FOR each candidate IN first_pass_cache[image.id]:
            low_score = 0.08 <= candidate.score < 0.25

            border_case = (
                candidate.source == "tile"
                AND distance_to_internal_tile_border(candidate) <= 96
                AND 0.10 <= candidate.score < 0.50
            )

            IF NOT low_score AND NOT border_case:
                CONTINUE

            roi_origin = CENTERED_ORIGIN(
                center(candidate.box),
                crop_size = 1024,
                image_bounds = image.size
            )

            IF roi_origin is already an original SAHI tile origin:
                CONTINUE

            priority = candidate.score + (2.0 IF border_case ELSE 1.0)
            triggers.APPEND({origin: roi_origin, priority: priority})

        triggers = SORT_DESCENDING(triggers, key = priority)
        triggers = DEDUPLICATE_ORIGINS(triggers, distance = 192)
        triggers = TAKE_FIRST(triggers, 4)

        refine_candidates = []
        FOR each trigger IN triggers:
            roi = CROP(image, trigger.origin, 1024, 1024)
            local_detections = detector.predict(
                roi,
                imgsz = 1024,
                conf  = CANDIDATE_FLOOR
            )
            refine_candidates += MAP_BOXES_TO_GLOBAL(local_detections, trigger.origin)

        refinement_cache[image.id] = refine_candidates

    # ------------------------------------------------------------
    # Stage C: 构建 V3E anchor
    # ------------------------------------------------------------
    v3e_anchor = []

    FOR each image IN images:
        selected = []

        selected += FILTER(
            first_pass_cache[image.id],
            score >= V3_FIRST_THRESHOLD
        )

        selected += FILTER(
            refinement_cache[image.id],
            score >= V3_REFINE_THRESHOLD
        )

        image_anchor = CLASSWISE_NMS(selected, iou = NMS_IOU)
        v3e_anchor += image_anchor

    # 当前运行得到 2570 个 V3E anchor

    # ------------------------------------------------------------
    # Stage D: 从首轮 SAHI 中提取未解释弱证据
    # ------------------------------------------------------------
    weak_region_plan = {}

    FOR each image IN images:
        weak = []

        FOR each candidate IN first_pass_cache[image.id]:
            IF NOT (WEAK_MIN <= candidate.score < WEAK_MAX):
                CONTINUE

            same_class_anchors = FIND(
                v3e_anchor,
                image_id == image.id AND class == candidate.class
            )

            IF EXISTS anchor IN same_class_anchors
               WHERE IoU(candidate.box, anchor.box) >= ANCHOR_EXPLAIN_IOU:
                CONTINUE

            weak.APPEND(candidate)

        class_aware_clusters = []

        FOR each candidate IN SORT_DESCENDING(weak, key = score):
            cluster = FIND_FIRST class_aware_clusters WHERE:
                cluster.class == candidate.class
                AND (
                    IoU(candidate.box, cluster.representative.box) >= CLUSTER_IOU
                    OR IoM(candidate.box, cluster.representative.box) >= CLUSTER_IOM
                )

            IF cluster exists:
                cluster.members.APPEND(candidate)
            ELSE:
                class_aware_clusters.APPEND(
                    NEW_CLUSTER(representative = candidate)
                )

        FOR each cluster IN class_aware_clusters:
            cluster.support = COUNT_UNIQUE(
                member.source,
                member.tile_origin
            )
            cluster.priority = cluster.max_score
                             + 0.025 * MIN(3, cluster.support - 1)

        class_aware_clusters = SORT_DESCENDING(
            class_aware_clusters,
            key = priority
        )

        regions = []
        FOR each cluster IN class_aware_clusters:
            nearest_region = FIND_NEAREST_REGION(
                regions,
                center(cluster.representative.box),
                max_distance = REGION_DEDUPE_DISTANCE
            )

            IF nearest_region exists:
                nearest_region.seeds.APPEND(cluster.representative)
            ELSE IF LENGTH(regions) < MAX_WEAK_REGIONS:
                regions.APPEND(NEW_REGION(cluster.representative))

        weak_region_plan[image.id] = regions

    # ------------------------------------------------------------
    # Stage E: 640 zoom + 1280 context 双视野复检
    # ------------------------------------------------------------
    v4_candidates = []

    FOR each image IN images:
        FOR each region IN weak_region_plan[image.id]:
            FOR each crop_size IN [ZOOM_SIZE, CONTEXT_SIZE]:
                crop_origin = CENTERED_ORIGIN(
                    region.center,
                    crop_size,
                    image.size
                )
                crop = CROP(image, crop_origin, crop_size, crop_size)

                local_detections = detector.predict(
                    crop,
                    imgsz = MODEL_IMGSZ,
                    conf  = CANDIDATE_FLOOR
                )

                global_detections = MAP_BOXES_TO_GLOBAL(
                    local_detections,
                    crop_origin
                )

                FOR each detection IN global_detections:
                    IF detection.score < V4_ACCEPT_THRESHOLD:
                        CONTINUE

                    confirmed = FALSE
                    FOR each seed IN region.seeds:
                        IF detection.class != seed.class:
                            CONTINUE

                        IF IoU(detection.box, seed.box) >= CONFIRM_IOU
                           OR IoM(detection.box, seed.box) >= CONFIRM_IOM:
                            confirmed = TRUE
                            BREAK

                    IF confirmed:
                        v4_candidates.APPEND(detection)

    v4_candidates = CLASSWISE_NMS_PER_IMAGE(
        v4_candidates,
        iou = NMS_IOU
    )

    # ------------------------------------------------------------
    # Stage F: anchor-preserving safe-add
    # ------------------------------------------------------------
    output = COPY(v3e_anchor)
    spatial_index = BUILD_INDEX(output, key = [image_id, class])

    FOR each candidate IN v4_candidates:
        same_class_boxes = spatial_index[candidate.image_id, candidate.class]

        IF EXISTS box IN same_class_boxes
           WHERE IoU(candidate.box, box) >= NOVEL_IOU:
            CONTINUE

        output.APPEND(candidate)
        same_class_boxes.APPEND(candidate.box)

    ASSERT output[0 : LENGTH(v3e_anchor)] == v3e_anchor

    RETURN output
    # 当前运行：2570 个 V3E anchor + 255 个新框 = 2825 个 V4E 框
```

其中：

```text
IoM(A, B) = area(A intersection B) / min(area(A), area(B))
```

IoM 用于处理大小差异明显但存在包含关系的框；class-wise NMS 和 safe-add 则始终只在同一类别内部进行。
