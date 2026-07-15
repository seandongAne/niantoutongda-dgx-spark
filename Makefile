# 手册 §3.2 标准入口。未实现的入口显式失败,不假装可用。
.PHONY: test deploy fixture-check demo benchmark

test:
	.venv/bin/python -m pytest backend/tests -q

deploy:
	./scripts/deploy.sh

fixture-check:
	@echo "NOT IMPLEMENTED: G0 夹具尚未冻结" && exit 1

demo:
	@echo "NOT IMPLEMENTED: 主链尚未接通" && exit 1

benchmark:
	@echo "NOT IMPLEMENTED: 探针/评测尚未就绪" && exit 1
