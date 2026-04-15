# TODO

## 2026-04-15

### Completed

- [x] Demo showcase branch final verification and path normalization fix
- [x] Push `feature/demo-showcase-report` after fixing local Git push/auth state

### Completed

- [x] Write the CI hardening design draft
  - Output: `docs/superpowers/specs/2026-04-15-ci-hardening-design.md`

### In Progress

- [x] Add a demo showcase smoke check to GitHub Actions CI
- [x] Add a FastAPI smoke check to GitHub Actions CI
- [x] Evaluate whether Docker build smoke should run on every PR
  - Decision: do not run Docker build smoke on every PR for now
  - Follow-up: consider a separate `docker-smoke` job for `develop`, release branches, or manual dispatch

### Next

- [ ] Prepare the PR from `feature/demo-showcase-report` to `develop`
- [ ] Re-review `README.md` and `README.ko.md` against the current showcase behavior
- [ ] Decide the next post-showcase priority: threshold tuning, showcase UX polish, or benchmark/report polish
