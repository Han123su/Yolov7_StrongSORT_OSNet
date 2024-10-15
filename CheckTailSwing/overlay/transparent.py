import cv2
import os
import glob

path = glob.glob("Source/*.jpg")
image = []
i=0
print(path)
for img in path:
    i=i+1
    img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)         # 因為是 jpg，要轉換顏色為 BGRA
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        # 新增 gray 變數為轉換成灰階的圖片
    h = img.shape[0]     # 取得圖片高度
    w = img.shape[1]     # 取得圖片寬度

# 依序取出圖片中每個像素
    for x in range(w):
        for y in range(h):
            if gray[y, x]>210:
                img[y, x, 3] = 255 - gray[y, x]
                # 如果該像素的灰階度大於 210，調整該像素的透明度
                # 使用 255 - gray[y, x] 可以將一些邊緣的像素變成半透明，避免太過鋸齒的邊緣

    cv2.imwrite('temp.png', img)    # 存檔儲存為 png
    cv2.waitKey(0)                        # 按下任意鍵停止
    cv2.destroyAllWindows()

    from PIL import Image
   
    img = Image.open('temp.png')
    img = img.convert("RGBA") # 轉換取得資訊
    pixdata = img.load()
  
    for y in range(img.size[1 ]):
        for x in range(img.size[0]):
            if pixdata[x, y][0] < 50 and pixdata[x, y][1] < 50 and pixdata[x, y][2] < 50 and pixdata[x, y][3] > 220 :
                pixdata[x, y] = (255, 255, 255 , 0)
    img.save('Result/result'+str(i)+'.png')
os.remove('temp.png')
