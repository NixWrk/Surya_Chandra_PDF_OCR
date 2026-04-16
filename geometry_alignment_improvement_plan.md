---
name: geometry_alignment_improvement_plan
description: Plan to improve text-to-PDF geometry alignment in hybrid mode.
type: project
---

**Goal**: Increase the precision of document boundary detection and warping in `DETECTOR_BACKEND_CV_HYBRID` mode to ensure that OCR text coordinates align perfectly with the PDF output, minimizing edge cropping or distortion.

**Context**: 
The current "hybrid" detector uses a heuristic scoring system (`_contour_score`) based on coverage and fill ratio across multiple OpenCV detection strategies (Quad, Hough, MinRect). While robust for finding *a* document, it lacks sub-pixel precision and does enough not account for the potential loss of content at the extreme edges of the warped frame.

**Proposed Approach**:

### 1. Sub-pixel Corner Refinement
The current `warp_perspective_from_points` uses points derived from integer-based contour approximations.
*   **Action**: Implement a refinement step using `cv2.cornerSubPix` on the detected 4-point quadrilaterals before the perspective transform is calculated.
*   **Target File**: `src/uniscan/core/scanner_adapter.py` (within the detection loop) or `src/uniscan/core/geometry.py`.

### 2. Edge-Margin Verification Pass
The current pipeline warps the image and then processes it. If the warp is too "tight," text at the boundary is lost.
*   **Action**: Introduce a "margin expansion" check. After an initial detection, attempt to expand the detected quad slightly (e.g., by 2-5%) and re-evaluate if this improves the `_contour_score` or preserves more content without introducing background noise.
*   **Target File**: `src/uniscan/core/scanner_adapter.py`.

### 3. Enhanced Feature-Based Detection (Optional/Secondary)
The Hough and Contour methods are sensitive to lighting/shadows.
*   **Action**: Add a lightweight feature-based detector using ORB descriptors to identify high-contrast corners, which can serve as a tie-breaker or a fourth candidate in the hybrid selection process.
*   **Target File**: `src/uniscan/core/scanner_adapter.py`.

### 4. Coordinate Integrity Check (Validation)
Ensure that the transformation matrix does not result in extreme scaling at the corners which could degrade OCR quality.
*   **Action**: Add a check to the `ScanOutput` generation to flag transformations with extremely high distortion ratios.

**Critical Files to Modify**:
- `src/uniscan/core/scanner_adapter.py`: To implement refinement and expansion logic in the detector loop.
- `src/unisun/core/geometry.py`: To add sub-pixel utility functions if needed.

**Verification Plan**:
1.  **Unit Tests**: Create test cases with known 4-point quadrilaterals (including sub-pixel offsets) to verify `cornerSubPix` integration.
2.  **Regression Testing**: Run the existing pipeline on a set of "difficult" images (shadowed, low contrast) and compare the `warped` output quality/coverage against the current implementation.
3.  **Visual Inspection**: Use the web-GUI (if available) to visually inspect if text previously clipped at edges is now preserved in the warped PNGs.
