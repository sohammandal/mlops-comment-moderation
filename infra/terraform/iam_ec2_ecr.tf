# EC2 instance role with ECR read
data "aws_iam_policy_document" "ec2_trust" {
  statement {
    effect = "Allow"
    principals { 
        type = "Service"
        identifiers = ["ec2.amazonaws.com"] 
    }
    actions   = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role" "ec2_ecr_read_role" {
  name               = "mlops-ec2-ecr-read"
  assume_role_policy = data.aws_iam_policy_document.ec2_trust.json
}

resource "aws_iam_role_policy_attachment" "ec2_ecr_read_attach" {
  role       = aws_iam_role.ec2_ecr_read_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_instance_profile" "ec2_ecr_read_profile" {
  name = "mlops-ec2-ecr-read"
  role = aws_iam_role.ec2_ecr_read_role.name
}
