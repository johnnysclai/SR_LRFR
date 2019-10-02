# sface + bicubic
python lfw_verification.py --fnet=sface --fnet_pth=../pretrained/sface.pth --isSR=False --down_factor=16;
python lfw_verification.py --fnet=sface --fnet_pth=../pretrained/sface.pth --isSR=False --down_factor=8;
python lfw_verification.py --fnet=sface --fnet_pth=../pretrained/sface.pth --isSR=False --down_factor=4;
python lfw_verification.py --fnet=sface --fnet_pth=../pretrained/sface.pth --isSR=False --down_factor=1;
# sface + edsr_baseline
python lfw_verification.py --fnet=sface --fnet_pth=../pretrained/sface.pth --isSR=True --srnet=edsr --srnet_pth=../pretrained/edsr_baseline.pth --down_factor=16;
python lfw_verification.py --fnet=sface --fnet_pth=../pretrained/sface.pth --isSR=True --srnet=edsr --srnet_pth=../pretrained/edsr_baseline.pth --down_factor=8;
python lfw_verification.py --fnet=sface --fnet_pth=../pretrained/sface.pth --isSR=True --srnet=edsr --srnet_pth=../pretrained/edsr_baseline.pth --down_factor=4;
# sface + edsr_lambda0.5
python lfw_verification.py --fnet=sface --fnet_pth=../pretrained/sface.pth --isSR=True --srnet=edsr --srnet_pth=../pretrained/edsr_lambda0.5.pth --down_factor=16;
python lfw_verification.py --fnet=sface --fnet_pth=../pretrained/sface.pth --isSR=True --srnet=edsr --srnet_pth=../pretrained/edsr_lambda0.5.pth --down_factor=8;
python lfw_verification.py --fnet=sface --fnet_pth=../pretrained/sface.pth --isSR=True --srnet=edsr --srnet_pth=../pretrained/edsr_lambda0.5.pth --down_factor=4;