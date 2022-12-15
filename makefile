requeriments:
	pip install -r requeriments.txt --quiet

q_learning:
	python3 -m q_learning/InitializeQvalues
	python3 -m q_learning/snake

deep_q_learning:
	python3 -m deep_q_learning/agent

policy_gradient:
	python3 -m policy_gradient/snake