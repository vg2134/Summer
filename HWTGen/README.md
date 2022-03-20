# HWTGen
## NYU-Information Technology Projects-2021 Summer-NYPL Group
### Group memeber: Anci Hu, Ke Shi, Yuze Gong
### Professor: Jean-Claude Franchitti
### Teaching Assistant: Joanna Gilberti
HWTGen, a machine-driven and crowd-rescoured meta-learning system to recognize cursive handwriting collections.

Python version```python==3.6```

## Tutorial
### Step1.Clone the project to local
```
git clone https://github.com/AmerGong/NYPL-HTRModel-NYU-ITP2021summer.git
```

### Step2. Set Up & Configure MySQL and Redis

Use [docker](https://www.docker.com/) to set up and configure MySQL and Redis. Here are the docker documentation for MySQL and Redis:

- [MySQL](https://hub.docker.com/_/mysql)

- [Redis](https://hub.docker.com/_/redis/)

But remember that the configuration information matches the env file.

### Step3. Install requirement.txt
```
pip install -r requirements.txt
```

### Step4. Install warp-ctc
You will need to install the following libraries from source. [warp-ctc](https://github.com/SeanNaren/warp-ctc) is needed for training.

`WARP_CTC_PATH` should be set to the location of a built WarpCTC
(i.e. `libwarpctc.so`).  This defaults to `../build`, so from within a
new warp-ctc clone you could build WarpCTC like this:

```bash
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc
mkdir build; cd build
cmake ..
make
```

Now install the bindings:
```bash
cd pytorch_binding
python setup.py install
```

If you try the above and get a dlopen error on OSX with anaconda3 (as recommended by pytorch):
```bash
cd ../pytorch_binding
python setup.py install
```

### Step5. Install mysqlclient
"mysqlclient" is a python packet. 

```pip install mysqlclient```

Because some machines cannot install mysqlclient. Then you will need to install and configure [PyMySQL](https://github.com/PyMySQL/PyMySQL/)

### Step6. Run
You need to run

```
python manage.py makemigrations
```

```
python manage.py migrate
```

```
python manage.py runserver
```

At the same time, open another terminal. Run:
```
python manage.py celery worker -c 4 --loglevel=info
```

Copy the IP address given to your browser. You can start using HWTGen!
