# About
The src folder contains functions that you can use to transfer data to and from Microsoft SharePoint Online.

# Usage

The code uses Office365-REST-Python-Client library to connect to Microsoft SharePoint using Python. 


1. [Installation](#Installation)
2. [Sharepoint Access Authentication](#Sharepoint-Access-Authentication)
3. [Functions](#Functions) 


# Installation

Use pip:

```
pip install Office365-REST-Python-Client
```


# SharePoint Access Authentication

Microsoft SharePoint Online source connection uses app-based authentication (Client ID and Client Secret) instead of user-based authentication (Username and Password).
Follow the instructions [here](https://support.google.com/workspacemigrate/answer/9545544?hl=en#:~:text=In%20your%20SharePoint%20Online%20tenant,aspx%20page.&text=Next%20to%20Client%20ID%20and,in%20your%20SharePoint%20Online%20environment) to create app-based credentials for SharePoint Online.

For authentication, we then need our access credentials (CLIENT_ID and CLIENT_SECRET) which can be stored in a .env file at project level.

    CLIENT_ID = "<CLIENTID>"
    CLIENT_SECRET = "<CLIENT_SECRET>"
    SITE_URL = "https://<CompanyUrl>.sharepoint.com/sites/<site-name>"

    ctx = ClientContext(SITE_URL).with_credentials(
            ClientCredential(CLIENT_ID, CLIENT_SECRET)
            )


# Functions

We have included four functions to:
- Download a file from SharePoint: download_from_sharepoint()
- Upload a file to SharePoint: upload_file_to_sharepoint()
- Write a pandas dataframe directly to SharePoint: upload_dataframe_to_sharepoint()
- Read a file directly from Sharepoint: read_sharepoint_file()



### References

[Office365-REST-Python-Client](https://github.com/vgrem/Office365-REST-Python-Client)

