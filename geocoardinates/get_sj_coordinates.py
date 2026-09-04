import coordinates as location


def main():
    coordinates = location.get_coordinates(city="San Jose")

    if coordinates:
        latitude, longitude = coordinates
        print(f"San Jose: latitude={latitude}, longitude={longitude}")
    else:
        print(f"Sorry, I don't know the coordinates for San Jose.")

if __name__ == "__main__":
    main()