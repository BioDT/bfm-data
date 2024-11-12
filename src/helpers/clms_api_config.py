# src/helpers/clms_api_config.py

import json
import time

import jwt
import requests


class CopernicusLandConfigurator:
    """
    A class to manage Copernicus Land Monitoring System API authentication using JWT and tokens.
    """

    @staticmethod
    def load_service_key(key_file: str) -> dict:
        """
        Loads the service key from the provided JSON file.

        Args:
            key_file (str): The file path to the service key JSON file.

        Returns:
            dict: A dictionary containing the service key data.
        """
        with open(key_file, "r") as f:
            return json.load(f)

    @staticmethod
    def create_jwt_token(service_key: dict) -> str:
        """
        Creates a JWT token using the stored private key and service key.

        Args:
            service_key (dict): A dictionary containing the service key details,
                                including 'private_key', 'client_id', 'user_id', and 'token_uri'.

        Returns:
            str: A signed JWT token.
        """
        private_key = service_key["private_key"].encode("utf-8")

        claim_set = {
            "iss": service_key["client_id"],
            "sub": service_key["user_id"],
            "aud": service_key["token_uri"],
            "iat": int(time.time()),
            "exp": int(time.time() + 3600),
        }

        jwt_token = jwt.encode(claim_set, private_key, algorithm="RS256")
        return jwt_token

    @staticmethod
    def get_access_token(service_key: dict, jwt_token: str) -> str:
        """
        Requests an access token using the JWT token and service key.

        Args:
            service_key (dict): A dictionary containing the service key details,
                                including 'token_uri' where the request will be sent.
            jwt_token (str): A signed JWT token to authenticate the request.

        Returns:
            str: The access token if the request is successful, otherwise None.
        """
        token_url = service_key["token_uri"]

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }

        data = {
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": jwt_token,
        }

        response = requests.post(token_url, headers=headers, data=data)

        if response.status_code == 200:
            return response.json()["access_token"]
        else:
            print(f"Failed to obtain access token. Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    def make_authenticated_request(
        self, url: str, access_token: str
    ) -> requests.Response:
        """
        Makes an authenticated API request using the access token.
        Retries the request if the token is expired.

        Args:
            url (str): The API endpoint to make the request to.
            access_token (str): The access token to authenticate the request.
            service_key (dict): A dictionary containing the service key details,
                                including 'token_uri' where the request will be sent.
            jwt_token (str): A signed JWT token to authenticate the request.
            retries (int): The number of times to retry if the token is expired (default: 1).

        Returns:
            requests.Response: The HTTP response object from the API request.
        """
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }

        response = requests.get(url, headers=headers)
        return response
