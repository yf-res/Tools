# Cross-Cloud Photo Search Agent — OAuth Setup & FAISS Persistence (Add-On)

This add-on extends the original guide with **ready-to-run setup scripts** for:
- Google Photos OAuth (installed app flow)
- Microsoft Graph / OneDrive OAuth (device code via MSAL)
- FAISS index persistence (save/load) + rows metadata storage

> All examples are in **Python** and designed to be pasted into your project.

---

## 0) Prereqs & Environment

**Install deps**
```bash
pip install google-auth google-auth-oauthlib google-auth-httplib2 \
            msal requests requests-oauthlib faiss-cpu numpy python-dotenv
```


