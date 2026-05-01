# list available com ports


if __name__ == "__main__":
    import serial.tools.list_ports

    print("Available Ports:")
    ports = serial.tools.list_ports.comports()
    for port, desc, hwid in sorted(ports):
        print("{}: {} [{}]".format(port, desc, hwid))
