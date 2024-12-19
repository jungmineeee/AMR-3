import numpy as np
import cv2 as cv
import glob

# 체스보드 코너 감지 설정
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 체스보드 패턴의 내부 코너 크기 (6 x 8)
pattern_size = (6, 8)

# 체스보드의 실제 3D 좌표 준비
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# 3D 좌표와 2D 좌표를 저장할 리스트
objpoints = []  # 실제 3D 좌표
imgpoints = []  # 감지된 2D 좌표

# 체스보드 이미지 파일 읽기
images = glob.glob('*.jpg')

# 체스보드 코너 감지
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # 체스보드 코너 찾기
    ret, corners = cv.findChessboardCorners(gray, pattern_size, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
    
    if ret:
        objpoints.append(objp)  # 3D 좌표 저장
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)  # 2D 좌표 저장
        
        # 코너 그리기 및 시각화
        # cv.drawChessboardCorners(img, pattern_size, corners2, ret)
        #cv.imshow('Chessboard Corners', img)
        #cv.waitKey(500)

cv.destroyAllWindows()

# 캘리브레이션 수행
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 내부 매트릭스와 왜곡 계수 출력
print("Camera Matrix (Intrinsic Parameters):\n", mtx)
print("Distortion Coefficients:\n", dist)

# 원본 이미지 읽기
test_image = 'Fisheye1_15.jpg'
img = cv.imread(test_image)
h, w = img.shape[:2]
print(h, w)

# 새로운 카메라 매트릭스 계산 (ROI 포함)
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# 1. 왜곡 보정 방법: cv.undistort
dst1 = cv.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst1 = dst1[y:y+h, x:x+w]  # ROI 적용
print(dst1.shape)
cv.imwrite('undistorted1.png', dst1)  # 보정된 이미지 저장

# # 2. 왜곡 보정 방법: cv.remap
# mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
# dst2 = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# dst2 = dst2[y:y+h, x:x+w]  # ROI 적용
# cv.imwrite('undistorted2.png', dst2)  # 보정된 이미지 저장

# 결과 확인
cv.imshow('Original Image', img)
cv.imshow('Undistorted Image 1 (cv.undistort)', dst1)
#cv.imshow('Undistorted Image 2 (cv.remap)', dst2)
cv.waitKey(0)
cv.destroyAllWindows()