@echo off
title Image Labeling System - Phone Access (LAN)
REM Opens the server to the local network so a phone or tablet on the
REM same Wi-Fi can use it. Same logic as start.bat, plus --lan.
REM Windows may ask for firewall permission the first time: click Allow.
call "%~dp0start.bat" --lan
