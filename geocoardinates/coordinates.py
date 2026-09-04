"""
Program: City Coordinates Lookup

Description:
    This program accepts a city name from the user and returns its
    corresponding latitude and longitude. 

Supported Cities:
    - San Jose
    - San Francisco
    - New York
    - Los Angeles

If the city is not supported, the program displays an appropriate
message to the user.
"""


def get_coordinates(city):
    """
    Return the latitude and longitude for a supported city.

    Args:
        city (str): The name of the city to look up.

    Returns:
        tuple: A tuple containing latitude and longitude if the city
               is supported; otherwise, None.
    """

    # Convert the city name to lowercase so the lookup is case-insensitive.
    match city.lower():
        case "san jose":
            return 37.3382, -121.8863
        case "san francisco":
            return 37.7749, -122.4194
        case "new york":
            return 40.7128, -74.0060
        case "los angeles":
            return 34.0522, -118.2437
        case _:
            # Return None when the city is not in the supported list.
            return None


def main():
    city = input("Enter a city: ")

    coordinates = get_coordinates(city)

    if coordinates:
        latitude, longitude = coordinates
        print(f"{city}: latitude={latitude}, longitude={longitude}")
    else:
        print(f"Sorry, I don't know the coordinates for {city}.")


if __name__ == "__main__":
    main()

