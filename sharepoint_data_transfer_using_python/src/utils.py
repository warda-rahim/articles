from office365.sharepoint.client_context import ClientContext
from office365.runtime.auth.client_credential import ClientCredential
from src.config import SITE_URL, CLIENT_ID, CLIENT_SECRET


def connect_to_sharepoint(site_url:str=SITE_URL, client_id:str=CLIENT_ID, client_secret:str=CLIENT_SECRET):
    """App-based authentication for Microsoft Sharepoint Online connection.
    Args:
        site_url (str): sharepoint site url (example: "https://<CompanyUrl>.sharepoint.com/sites/<site-name>")
        client_id (str): GUID for sharepoint app 
        client_secret (str): password for sharepoint app
    Returns:
        office365.sharepoint.client_context.ClientContext
    """

    ctx = ClientContext(site_url).with_credentials(
            ClientCredential(client_id, client_secret)
            )

    return ctx



