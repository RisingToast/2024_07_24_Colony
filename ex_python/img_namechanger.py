import os

def rename_images(directory):
    # 디렉토리 내 모든 파일 목록 가져오기
    files = os.listdir(directory)
    
    # 이미지 파일만 필터링 (예: jpg, png)
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    images = [f for f in files if f.lower().endswith(image_extensions)]
    
    # 이미지 파일을 번호 순서대로 정렬
    images.sort()
    
    for index, image in enumerate(images):
        # 새로운 파일명 생성
        new_name = f"{index + 851:03d}{os.path.splitext(image)[1]}"
        
        # 기존 파일 경로와 새로운 파일 경로
        old_path = os.path.join(directory, image)
        new_path = os.path.join(directory, new_name)
        
        # 파일 이름 변경
        os.rename(old_path, new_path)
        print(f"Renamed '{image}' to '{new_name}'")

# 사용할 디렉토리 경로를 지정하세요.
directory_path = 'C:/kkt/2024_07_24_Colony/Resize_img/re.50.Streptococcus agalactiae'
rename_images(directory_path)
