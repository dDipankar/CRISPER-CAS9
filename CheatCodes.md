To know the disk size in centos: **df -H --output=size,used,avail**

To know the RAM size in centos: **cat /proc/meminfo**, **free -m**

To know the list of python programs: **top -p $(pgrep -d',' python)**

To change the priority of an existing process just do 

[renice] -p [process_id]:

For Example:
renice 10 -p 21827


GPU info:
nvidia-smi

To find files that contain a specific word in a directory: grep -r 'sequences.txt' .

copy : cp sequences_30bp.txt /home/csgrads/dbais001/CRISP-CAS9/yeast_bact/yeast_data/Group3/

