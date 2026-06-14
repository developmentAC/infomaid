# Security Advisory - ChromaDB Vulnerability

## Active Security Issue

### CVE-2026-45829: ChromaDB Pre-authentication Code Injection

**Status:** No patch available yet  
**Severity:** Critical (CVSS 9.3/10)  
**Date Identified:** 2026-05-14  
**Last Checked:** 2026-06-14

#### Vulnerability Details
- **Affected Package:** chromadb
- **Affected Versions:** >= 1.0.0, <= 1.5.9
- **Current Project Version:** 1.5.9
- **Patched Version:** None available
- **Advisory:** [GHSA-f4j7-r4q5-qw2c](https://github.com/advisories/GHSA-f4j7-r4q5-qw2c)
- **CVE:** [CVE-2026-45829](https://nvd.nist.gov/vuln/detail/CVE-2026-45829)

#### Description
A pre-authentication code injection vulnerability allows an unauthenticated attacker to execute arbitrary code on the server by sending a malicious model repository with `trust_remote_code=true` to the `/api/v2/tenants/{tenant}/databases/{db}/collections` API endpoint.

#### Risk Assessment for This Project: LOW ✓

**Why the risk is LOW for Infomaid:**
1. **Embedded Mode Only:** This project uses ChromaDB in embedded/local mode (`persist_directory="chroma"`)
2. **No Server Exposure:** Not running ChromaDB as a network-accessible server
3. **No API Usage:** Does not utilize ChromaDB's HTTP API endpoints
4. **Local Application:** Designed for local development and use only

**The vulnerability targets ChromaDB server API endpoints, which this project does not expose.**

#### Mitigation Measures

**Current Protections:**
- ✅ Using ChromaDB in embedded/client mode only
- ✅ No network server running
- ✅ Local file-based vector storage
- ✅ No external API exposure

**Additional Recommendations:**
1. **Do NOT run ChromaDB as a server** (avoid `chroma run` or server mode)
2. **Keep application local** - do not expose to untrusted networks
3. **Monitor for updates** - check regularly for patched versions
4. **Review dependencies** - run `poetry show chromadb` periodically

#### Monitoring Plan

Check for updates weekly using:
```bash
# Check current version
poetry show chromadb

# Check for newer versions
pip index versions chromadb | head -20

# Update when patch is available
poetry update chromadb
```

#### Update History
- **2026-06-14:** Initial security advisory created, risk assessed as LOW for this project
- **Next Review:** Check for patches by 2026-06-21

#### References
- [GitHub Advisory Database](https://github.com/advisories/GHSA-f4j7-r4q5-qw2c)
- [ChromaDB Issue #6717](https://github.com/chroma-core/chroma/issues/6717)
- [HiddenLayer Research](https://www.hiddenlayer.com/research/chromatoast-served-pre-auth)
- [NVD CVE-2026-45829](https://nvd.nist.gov/vuln/detail/CVE-2026-45829)

#### Contact
For security concerns, contact: obonhamcarter@allegheny.edu

---

## General Security Best Practices

1. **Keep Dependencies Updated:** Regularly update all dependencies when security patches are available
2. **Monitor Security Advisories:** Check GitHub Dependabot alerts regularly
3. **Local Development:** Continue using the application locally without network exposure
4. **Environment Isolation:** Use virtual environments (Poetry) to manage dependencies
5. **Review Code:** Be cautious when adding new network-facing features

---

**Last Updated:** 2026-06-14  
**Next Review Date:** 2026-06-21
