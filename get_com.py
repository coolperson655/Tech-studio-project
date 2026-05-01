# list available com ports


if __name__ == "__main__":
    import serial.tools.list_ports

    print("Available Ports:")
    ports = serial.tools.list_ports.comports()
    for port, desc, hwid in sorted(ports):
        print("{}: {} [{}]".format(port, desc, hwid))


    print("if ur on mac:")
    # Source - https://stackoverflow.com/a/26755442
    # Posted by tfeldmann
    # Retrieved 2026-05-01, License - CC BY-SA 3.0
    import glob
    glob.glob('/dev/tty.*')
