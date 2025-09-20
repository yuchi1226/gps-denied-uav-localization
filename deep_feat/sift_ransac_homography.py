import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

def get_param(img_batch, template_batch, image_sz):
    template = img_batch.squeeze(0).cpu().numpy()
    img = template_batch.squeeze(0).cpu().numpy()
    
    if template.shape[0] == 3:
        template = np.transpose(template, (1, 2, 0))
        img = np.transpose(img, (1, 2, 0))
        
        template = (template * 255).astype('uint8')
        img = (img * 255).astype('uint8')
    
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 使用 ORB 特徵檢測器替代 SIFT（SIFT 有專利問題）
    orb = cv2.ORB_create()
    
    kp1, des1 = orb.detectAndCompute(template_gray, None)
    kp2, des2 = orb.detectAndCompute(img_gray, None)
    
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        H_found = np.eye(3)
    else:
        # 使用 BFMatcher 進行特徵匹配
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < 4:
            H_found = np.eye(3)
        else:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            src_pts = src_pts - image_sz/2
            dst_pts = dst_pts - image_sz/2
            
            H_found, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
            
            if H_found is None:
                H_found = np.eye(3)
    
    H = torch.from_numpy(H_found).float()
    I = torch.eye(3, 3)
    
    p = H - I
    p = p.view(1, 9, 1)
    p = p[:, 0:8, :]
    
    if torch.cuda.is_available():
        return p.cuda()
    else:
        return p

if __name__ == "__main__":
    img = cv2.imread('../duck.jpg')
    img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    rows, cols, ch = img_color.shape
    
    pts1 = np.float32([[0,0],[0,rows],[cols,rows],[cols,0]])
    pts2 = np.float32([[0,0],[0,rows],[cols+200,rows],[cols+200,0]])
    
    H_gt = cv2.getPerspectiveTransform(pts1, pts2)
    
    print(H_gt)
    
    dst_img = cv2.warpPerspective(img_color, H_gt, (cols, rows))
    
    template_batch = torch.from_numpy(np.transpose(img_color, (2, 0, 1))).unsqueeze(0).float()
    img_batch = torch.from_numpy(np.transpose(dst_img, (2, 0, 1))).unsqueeze(0).float()
    
    print(get_param(img_batch, template_batch, cols))