requeriments:
	pip install -r requeriments.txt --quiet

q_learning:
	@python -m q_learning_tradicional.InitializeQvalues
	@python -m q_learning_tradicional.snake

deep_q_learning:
	@python -m deep_q_learning.agent

policy_gradient:
	@python -m policy_gradient.snake