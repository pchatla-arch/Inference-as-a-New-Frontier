# Hosting the package on GitHub

Suggested repository name: `inference-as-a-new-frontier`.

## Repository

Extract the ZIP locally. Create a new GitHub repository under the account and visibility you choose. Upload or commit the **contents** of the extracted `inference-as-a-new-frontier` directory, so `README.md`, `CITATION.cff`, `index.html`, and the `paper/` directory are at the repository root. Uploading only the ZIP will not expose the README, source tree, or citation metadata as intended.

A standard Git workflow is:

```bash
git init
git add .
git commit -m "Add inference systems paper and reproducibility companion"
git branch -M main
# Use the remote URL GitHub supplies for your newly created repository:
git remote add origin YOUR_REPOSITORY_REMOTE
git push -u origin main
```

`YOUR_REPOSITORY_REMOTE` is the only placeholder in this example. No remote has been created or accessed by this package. Use a fresh repository for these commands; an existing repository may require its normal branch and merge workflow.

## Optional GitHub Pages site

A self-contained `index.html` and `.nojekyll` are included. After uploading, open the repository's **Settings → Pages**, choose **Deploy from a branch**, select `main` and `/(root)`, and save. Use the Pages URL reported by GitHub rather than assuming a URL before deployment. Availability depends on the repository and account configuration. A Pages site can be public even when its backing repository is private; review publication scope before enabling it.

## Citation and release housekeeping

Keep `CITATION.cff` on the default branch. It uses `preferred-citation` to point to the technical manuscript rather than treating the paper as a software publication. Add the real repository URL after creation. Add a DOI only after a DOI has actually been issued. The package version is `1.1.0`; creating an optional release with that tag does not imply peer review.

Choose reuse terms before representing the project as openly licensed. `RIGHTS.md` intentionally records that no license has been chosen on the author's behalf.

## Official documentation

- Citation metadata: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files
- Pages publishing source: https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site

Documentation checked September 5, 2026.
