#!/usr/bin/env bash
# Idempotent EC2 instance role + instance profile.

set -euo pipefail
source "$(dirname "$0")/env.sh"

POLICY_DIR="$AWS_SETUP_DIR/policies"
TRUST=$(cat "$POLICY_DIR/ec2-instance-trust.json")
PERMS=$(sed -e "s|__ACCOUNT_ID__|$AWS_ACCOUNT_ID|g" \
            -e "s|__AWS_REGION__|$AWS_REGION|g" \
            "$POLICY_DIR/ec2-instance-permissions.json")

# Create role.
if aws iam get-role --role-name "$IAM_ROLE_INSTANCE" >/dev/null 2>&1; then
    log "Role $IAM_ROLE_INSTANCE already exists — refreshing trust policy"
    aws iam update-assume-role-policy \
        --role-name "$IAM_ROLE_INSTANCE" \
        --policy-document "$TRUST"
else
    log "Creating role $IAM_ROLE_INSTANCE"
    aws iam create-role \
        --role-name "$IAM_ROLE_INSTANCE" \
        --assume-role-policy-document "$TRUST" \
        --description "Attached to EC2 instance; reads SSM params and pulls ECR." \
        >/dev/null
fi

# Attach the AWS-managed SSM core policy (enables SSM agent + Session Manager).
log "Attaching AmazonSSMManagedInstanceCore"
aws iam attach-role-policy \
    --role-name "$IAM_ROLE_INSTANCE" \
    --policy-arn "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"

# Custom inline policy.
log "Putting inline policy ec2-rag-instance-permissions"
aws iam put-role-policy \
    --role-name "$IAM_ROLE_INSTANCE" \
    --policy-name "ec2-rag-instance-permissions" \
    --policy-document "$PERMS"

# Instance profile.
if aws iam get-instance-profile --instance-profile-name "$IAM_INSTANCE_PROFILE" >/dev/null 2>&1; then
    log "Instance profile $IAM_INSTANCE_PROFILE already exists"
else
    log "Creating instance profile $IAM_INSTANCE_PROFILE"
    aws iam create-instance-profile --instance-profile-name "$IAM_INSTANCE_PROFILE" >/dev/null
fi

# Add role to profile.
if aws iam get-instance-profile --instance-profile-name "$IAM_INSTANCE_PROFILE" \
    --query 'InstanceProfile.Roles[].RoleName' --output text \
    | grep -qw "$IAM_ROLE_INSTANCE"; then
    log "Role already attached to instance profile"
else
    log "Adding role to instance profile"
    aws iam add-role-to-instance-profile \
        --instance-profile-name "$IAM_INSTANCE_PROFILE" \
        --role-name "$IAM_ROLE_INSTANCE"
fi

ROLE_ARN=$(aws iam get-role --role-name "$IAM_ROLE_INSTANCE" --query Role.Arn --output text)
cache_set IAM_ROLE_INSTANCE_ARN "$ROLE_ARN"
log "Instance role ARN: $ROLE_ARN"
