from functionfile import run, cv2, saveFile, ocr

path = 'Program/Image Testing/Image/'
image_path = path + 'white.png'
image = cv2.imread(image_path)
crop, detect, _ = run(image)
saveFile(f'{path}/../output', detect, crop)
ocr(f'{path}/../output', crop)
# cv2.waitKey(0)
