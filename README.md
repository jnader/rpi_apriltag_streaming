# rpi_apriltag_streaming

## 1. Install requirements
```
sudo apt install hostapd
pip install flask
pip install imutils
pip install pupil-apriltags
```

## 2. Setup access point network and dhcpcd file to set `wlan0` interface.
```
sudo nano /etc/hostapd/hostapd.conf

interface=wlan0
ssid=pi_network
hw_mode=g
channel=6
auth_algs=1
wmm_enabled=0
```

```
sudo nano /etc/dhcpcd.conf

interface wlan0
static ip_address=192.168.0.10/24
```

## 3. Setup `~/.bashrc` for automatic script launch at boot
```
```

## 4. Test the project
