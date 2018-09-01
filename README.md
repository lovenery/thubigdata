# thubigdata 

## Build Dockerfile

```bash
git clone git@github.com:lovenery/thubigdata.git
cd thubigdata/
docker build -t thu_big_data_image .
docker run -it --rm thu_big_data_image
python3 main.py

# Detach mode
docker run -it -d --name thu_big_data_box thu_big_data_image
docker exec -it thu_big_data_box /bin/bash
```

## Packaging

```bash
# export
docker run -it -d --name thu_big_data_box thu_big_data_image
docker export thu_big_data_box > thu.tar

# import
cat thu.tar | docker import - thu_big_data_image
docker run -it --rm -w=/root thu_big_data_image /bin/bash
```

## Notes

```bash
docker run -it --rm --name thu_big_data_box -v `pwd`:/root lovenery/thu
pip install numpy==1.14.5 pandas matplotlib keras tensorflow sklearn xlrd wheel
pip3 install --upgrade setuptools
pip3 install -r requirements.txt

docker rm -f $(docker ps -a -q)
docker rmi -f $(docker images -q)
```

## Refs

- https://zhuanlan.zhihu.com/p/32501790
- https://medium.com/@daniel820710/%E5%88%A9%E7%94%A8keras%E5%BB%BA%E6%A7%8Blstm%E6%A8%A1%E5%9E%8B-%E4%BB%A5stock-prediction-%E7%82%BA%E4%BE%8B-1-67456e0a0b
