# OIDC provider for GitHub Actions
resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

# Trust policy
data "aws_iam_policy_document" "gha_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:sohammandal/mlops-comment-moderation:ref:refs/heads/main"]
    }
  }
}

resource "aws_iam_role" "gha_ecr_push" {
  name               = "mlops-github-actions-ecr-push"
  assume_role_policy = data.aws_iam_policy_document.gha_assume_role.json
}

# Allow push and pull on ECR repo
data "aws_iam_policy_document" "gha_ecr_policy_doc" {
  # ECR repo scoped permissions
  statement {
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:CompleteLayerUpload",
      "ecr:GetDownloadUrlForLayer",
      "ecr:UploadLayerPart",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:BatchGetImage",
      "ecr:DescribeRepositories",
      "ecr:DescribeImages",
      "ecr:ListImages"
    ]
    resources = [aws_ecr_repository.mlops.arn]
  }
  # GetAuthorizationToken must be "*"
  statement {
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "gha_ecr_policy" {
  name   = "mlops-github-actions-ecr"
  policy = data.aws_iam_policy_document.gha_ecr_policy_doc.json
}

resource "aws_iam_role_policy_attachment" "gha_ecr_attach" {
  role       = aws_iam_role.gha_ecr_push.name
  policy_arn = aws_iam_policy.gha_ecr_policy.arn
}

output "gha_role_arn" {
  value       = aws_iam_role.gha_ecr_push.arn
  description = "Use this ARN in GitHub Actions role-to-assume"
}
