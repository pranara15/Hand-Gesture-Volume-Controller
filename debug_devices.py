"""
Run this first to see ALL audio devices and their states.
Paste the output here so we can fix the filter.
"""
from pycaw.pycaw import AudioUtilities  # type: ignore

devices = AudioUtilities.GetAllDevices()
print(f"Total devices found: {len(devices)}\n")
for i, d in enumerate(devices):
    try:
        ep = d.EndpointVolume
        has_vol = ep is not None
    except Exception as e:
        has_vol = False
    print(f"[{i}] Name  : {d.FriendlyName}")
    print(f"     State : {d.state}")
    print(f"     HasVol: {has_vol}")
    print()
