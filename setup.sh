git clone https://github.com/jcperdomo/noise_filter.git
cd noise_filter

# use the given virtual env
pip install virtualenv
virtualenv env -p python3
source ./env/bin/activate
# this will take some time
pip3 install -r requirements.txt

# update database
python manage.py makemigrations noise_filter
python manage.py migrate
# this will take some time
python load_images.py

